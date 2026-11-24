'''
Simulation using the local-limit CBF + soft CLF with a chosen history length and a chosen
definition of "stability" (equilibrium):

  equilibrium="velocity"  velocity -> 0. Needs no reference; z holds raw plant-side quantities.
  equilibrium="tracking"  position error w.r.t. the LAST RECEIVED reference -> 0.
                          Assumes each forward packet carries (u, h), so the plant side holds
                          h_rx, which is exactly as stale as u_remote. z is in error coordinates:
                          every x entry has x_ref subtracted (same held reference for the whole
                          history, so the error dynamics stay the same linear model), u entries
                          have u_ref = 0 subtracted. When a new packet arrives x_ref jumps, so V
                          jumps; V only decreases between packets.

z(t) = [x(t), ..., x(t-n_x), u(t-1), ..., u(t-n_u)]
Per-step diagnostics are stored in simulate_run.last_log.
'''
from experiment_go import *
from building_block import *
from cbf_clf_local_hist import build_augmented, clf_params, History

import numpy as np

DEBUG = False


def simulate_run(N_s, h_seq, aoi_threshold_f, aoi_threshold_b, rng, make_cbf, get_u_cbf,
                 switching=True, seed=RNG_SEED, lossless=True, aoi_aware=1,
                 n_x=1, n_u=1, use_clf=True, equilibrium="velocity",
                 alpha=0.05, slack_weight=1.0):
    use_filter = make_cbf is not None and get_u_cbf is not None
    modes = (MODE_HUMAN, MODE_SAFE) if switching else (MODE_HUMAN,)

    ##### INITIAL STATE #####
    zeta = MODE_HUMAN
    Cp, Ap, Bp = C_plant[zeta], A_plant[zeta], B_plant[zeta]
    Cc, Ac, Bc, Dc = C_controller[zeta], A_controller[zeta], B_controller[zeta], D_controller[zeta]

    n, m = Ap.shape[0], Bp.shape[1]
    n_aug = n * (n_x + 1) + m * n_u

    clf = {}
    if use_filter and use_clf:
        clf = {zt: clf_params(A_plant[zt], B_plant[zt], n_x=n_x, n_u=n_u, equilibrium=equilibrium)
               for zt in modes}
    solve = make_cbf(n_aug, use_clf) if use_filter else None

    sigma_f, sigma_b = 1, 1
    aoi_f, aoi_b = 1, 1
    tracked_y_remote = np.zeros((2, N_s))

    x_controller = np.zeros((4, 2))
    x_plant = np.zeros((2, 2))
    h = h_seq[:, 0]
    x_plant[0, :] = h / Cp[0, 0]

    y_remote = Cp @ x_plant
    y_local = y_remote
    u_local = Cc @ x_controller + Dc @ (h - y_local)
    u_remote = u_local
    u_prev_eff = np.zeros((1, 2))
    h_rx = h.copy()          # last RECEIVED reference; updated together with u_remote

    hist = History(x_plant, n_x=n_x, n_u=n_u, m=m)
    log = {"V_now": [], "delta": [], "status": [], "fallback": [], "aoi": [],
           "pos_err_held": [], "pos_err_true": []}

    def build_z(Cp_now):
        z = hist.z()
        if equilibrium == "tracking":
            x_ref = np.zeros((n, 2))
            x_ref[0, :] = h_rx / Cp_now[0, 0]            # held reference, zero velocity
            z_ref = np.vstack([x_ref] * (n_x + 1) + [np.zeros((m, 2))] * n_u) # the current z state with the x_reference from h_rx (mever use just h only use h_rx here on the robto side)
            z = z - z_ref
        return z

    for t in range(N_s):
        # channel
        if lossless:
            sigma_f = sigma_b = 1
        else:
            sigma_f = int(rng.random() < (p1_f if sigma_f else 1 - p0_f))
            sigma_b = int(rng.random() < (p1_b if sigma_b else 1 - p0_b))

        aoi_f = aoi(sigma_f, aoi_f, AOI_CAP)
        aoi_b = aoi(sigma_b, aoi_b, AOI_CAP)

        # mode
        theta = get_theta(aoi_f, aoi_b, aoi_threshold_f, aoi_threshold_b)
        sf, sb, zeta = f_map(theta, aoi_threshold_f, aoi_threshold_b)
        if not switching:
            zeta = MODE_HUMAN

        Cp, Ap, Bp = C_plant[zeta], A_plant[zeta], B_plant[zeta]
        Cc, Ac, Bc, Dc = C_controller[zeta], A_controller[zeta], B_controller[zeta], D_controller[zeta]

        h = h_seq[:, t] #at each timestep h is always updated on the controller side but h_rx (h received) is only updated ont he remote robot side if the forward link worked. only use the h_rx 
        current_aoi = aoi_f if aoi_aware else 1

        # filtered (or raw) command
        if use_filter:
            kwargs = {}
            if use_clf:
                A_z, B_z, P, _ = clf[zeta]
                kwargs = dict(z=build_z(Cp), A_z=A_z, B_z=B_z, P=P, alpha=alpha, slack_weight=slack_weight)
            u_applied = get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, u_prev_eff,
                                  current_aoi=current_aoi, **kwargs)
            info = getattr(solve, "info", {})
            log["V_now"].append(info.get("V_now"))
            log["delta"].append(info.get("delta"))
            log["status"].append(info.get("status"))
            log["fallback"].append(info.get("fallback"))
            log["aoi"].append(current_aoi)
        else:
            u_applied = np.asarray(u_remote).reshape(1, 2)

        # tracking-error diagnostics (position, before the update): held vs true reference
        pos = x_plant[0, :]
        log["pos_err_held"].append(float(np.linalg.norm(pos - h_rx / Cp[0, 0])))
        log["pos_err_true"].append(float(np.linalg.norm(pos - h / Cp[0, 0])))

        # plant update
        x_next = Ap @ x_plant + Bp @ u_applied
        hist.push(x_next, u_applied)
        x_plant, u_prev_eff = x_next, u_applied

        y_remote = Cp @ x_plant
        y_local = y_remote if sigma_b else y_local
        tracked_y_remote[:, t] = y_remote.ravel()

        # operator side
        x_controller = Ac @ x_controller + Bc @ (h - y_local)
        u_local = Cc @ x_controller + Dc @ (h - y_local)
        if sigma_f:                                      # packet carries (u, h)
            u_remote = u_local
            h_rx = h.copy()

        if DEBUG:
            print("LOG u_local", u_local, "u_remote", u_remote)
            print(f"LOG aoi_f: {aoi_f}, aoi_b: {aoi_b}")

    simulate_run.last_log = log
    return tracked_y_remote