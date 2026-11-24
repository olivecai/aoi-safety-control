'''
Local-limit CBF (wall disc + AoI-shrinking local disc) with a CLF as a soft constraint via slack.

QP solved at each step, with y_next = Cp Ap x + Cp Bp u:

    min_{u, delta}  ||u - u_remote||^2 + slack_weight * delta
    s.t.  ||y_next||          <= rho          wall disc, centred at 0          (hard)
          ||y_next - y_t||    <= r(AoI)       local disc, centred at y_t       (hard)
          V(z_next)/s <= (1 - alpha) V(z)/s + delta,  delta >= 0               (CLF, soft)

V(z) = sum_k z_k^T P z_k = ||L z||^2 (P = L^T L), z_next = A_z z + B_z u, s = max(V(z), 1).
z = [x(t), x(t-1), u(t-1)] uses only plant-side quantities (no reference, Option A).

Only the local disc radius r depends on AoI. alpha and slack_weight must NOT depend on AoI.

Feasibility: u_hold = (y_t - Cp Ap x) / (Cp Bp) gives y_next = y_t, which lies inside both
discs (y_t is inside the wall disc by construction of rho, and r >= 0). So the CBF part is
always feasible, and the CLF slack means the whole QP is always feasible.
'''

from experiment_go import *

import cvxpy as cp
import numpy as np

N_AUG = 5          # z = [x(t), x(t-1), u(t-1)] for n=2, m=1
DEBUG = False      # set True to print per-step diagnostics


def make(use_clf=True):
    '''
    ret solve fn for the local-limit CBF (+ optional CLF)
    '''
    u = cp.Variable(2)                              # [u_x, u_y]
    u_r = cp.Parameter(2)                           # received command
    CpApxp = cp.Parameter(2)                        # Cp Ap x
    CpBp = cp.Parameter()                           # Cp Bp (scalar)
    rho = cp.Parameter(nonneg=True)                 # wall disc radius
    diff = cp.Parameter(2)                          # y_t - Cp Ap x
    r = cp.Parameter(nonneg=True)                   # local disc radius (AoI dependent)

    constraints = [
        cp.norm(CpApxp + CpBp * u) <= rho,          # wall disc, centred at 0
        cp.norm(CpBp * u - diff) <= r,              # local disc, centred at y_t  (== ||y_next - y_t|| <= r)
    ]
    objective = cp.sum_squares(u - u_r)

    if use_clf:
        delta = cp.Variable(nonneg=True)
        LAz_z = cp.Parameter((N_AUG, 2))            # L @ A_z @ z / sqrt(s)
        LBz = cp.Parameter((N_AUG, 1))              # L @ B_z / sqrt(s)
        V_bound = cp.Parameter(nonneg=True)         # (1 - alpha) * V_now / s
        slack_weight = cp.Parameter(nonneg=True)

        Lz_next = LAz_z + LBz @ cp.reshape(u, (1, 2), order="C")
        constraints += [cp.sum_squares(Lz_next) <= V_bound + delta]
        objective = objective + slack_weight * delta

    prob = cp.Problem(cp.Minimize(objective), constraints)

    def solve(u_r_v, CpApxp_v, CpBp_v, rho_v, diff_v, r_v,
              LAz_z_v=None, LBz_v=None, V_bound_v=None, slack_weight_v=None, V_now_v=None):
        u_r.value, CpApxp.value, CpBp.value, rho.value = u_r_v, CpApxp_v, CpBp_v, rho_v
        diff.value, r.value = diff_v, r_v
        if use_clf:
            LAz_z.value, LBz.value = LAz_z_v, LBz_v
            V_bound.value, slack_weight.value = V_bound_v, slack_weight_v

        tol = dict(tol_gap_abs=1e-6, tol_gap_rel=1e-6, tol_feas=1e-6)
        fallback = False
        try:
            prob.solve(solver=cp.CLARABEL, **tol)
        except cp.error.SolverError:
            fallback = True
            try:
                prob.solve(solver=cp.SCS)
            except cp.error.SolverError:
                solve.info = {"status": "solver_error", "delta": np.nan,
                              "V_now": V_now_v, "fallback": True}
                return None, False

        solve.info = {"status": prob.status,
                      "delta": float(delta.value) if use_clf and delta.value is not None else 0.0,
                      "V_now": V_now_v, "fallback": fallback}
        if u.value is None:
            return None, False
        return u.value, True

    solve.info = {}
    return solve


def local_radius(current_aoi, r_max=1.0, r_slope=0.5):
    '''AoI-dependent local disc radius: shrinks as AoI grows (0.5 * r_max at AoI = 1).'''
    return r_max / (1 + np.exp(r_slope * (current_aoi - 1)))


def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, u_prev, R=SAFE_SET_RADIUS,
              gamma=0.1, current_aoi=1, aoi_threshold=AOI_THRESHOLD,
              z=None, A_z=None, B_z=None, P=None, alpha=0.5, slack_weight=1.0):
    '''
    z is the plant-side augmented state (5x2). If z/A_z/B_z/P are None the CLF is skipped
    (requires make(use_clf=False)).
    '''
    CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-12:
        return np.asarray(u_remote).reshape(1, 2)

    y_t = np.asarray(y_remote).ravel()                  # current output, one entry per axis
    CpApxp = (Cp @ Ap @ x_plant).ravel()

    diff = y_t - CpApxp
    r = local_radius(current_aoi)
    rho = np.sqrt(max(R**2 - (1 - gamma) * (R**2 - y_t @ y_t), 0))

    if z is None:
        u, ok = solve(u_remote.ravel(), CpApxp, CpBp, rho, diff, r)
    else:
        P_sym = (P + P.T) / 2
        L = np.linalg.cholesky(P_sym).T                 # P = L^T L
        V_now = float(np.sum((L @ z) ** 2))             # sum_k z_k^T P z_k
        s = max(V_now, 1.0)
        LAz_z = (L @ (A_z @ z)) / np.sqrt(s)
        LBz = (L @ B_z) / np.sqrt(s)
        V_bound = (1 - alpha) * V_now / s
        u, ok = solve(u_remote.ravel(), CpApxp, CpBp, rho, diff, r,
                      LAz_z, LBz, V_bound, slack_weight, V_now)

    if not ok:
        u = diff / CpBp                                  # u_hold: y_next = y_t, inside both discs

    if DEBUG:
        y_next = CpApxp + CpBp * u
        print(f"AoI {current_aoi}, r {r:.4f}, |y_next|={np.linalg.norm(y_next):.4f} (rho {rho:.4f}), "
              f"|y_next-y_t|={np.linalg.norm(y_next - y_t):.4f}, info={getattr(solve, 'info', {})}")
    return np.asarray(u).reshape(1, 2)