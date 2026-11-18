'''
Standard CBF with CLF as a soft constraint via slack.

QP to solve at each step:
    min_{u, delta}  ||u - u_remote||^2 + slack_weight * delta
    s.t.  ||Cp Ap x + Cp Bp u|| <= rho_cbf                          (CBF, hard)
          V(z_next) / s <= (1 - alpha) * V(z) / s + delta,  delta >= 0   (CLF, soft)

with V(z) = sum over DOFs of z_k^T P z_k, z_next = A_z z + B_z u, and s = max(V(z), 1).
Dividing by s changes nothing about which u are allowed; it only keeps the numbers near 1
so the solver is well scaled, and makes delta a fraction of the current score.

P = L^T L (Cholesky), so z^T P z = ||L z||^2.
Neither alpha nor slack_weight may depend on current_aoi: only the CBF part uses AoI.
'''

from experiment_go import *

import cvxpy as cp
import numpy as np

N_AUG = 5   # dimension of z = [x(t), x(t-1), u(t-1)] for n=2, m=1


def make(use_clf=True):
    '''
    ret cbf(+clf) solve fn
    '''
    u_cbf = cp.Variable(2)                      # [u_x, u_y]
    
    u_r = cp.Parameter(2)                       # u_remote, the received command
    CpApxp = cp.Parameter(2)                    # C_p A_p x for each axis
    CpBp = cp.Parameter()                       # C_p B_p (scalar)
    rho = cp.Parameter(nonneg=True)             # CBF radius

    constraints = [cp.norm(CpApxp + CpBp * u_cbf) <= rho]
    objective = cp.sum_squares(u_cbf - u_r)

    if use_clf:
        delta = cp.Variable(nonneg=True)                 # CLF slack (fraction of current score)
        
        LAz_z = cp.Parameter((N_AUG, 2))                 # L @ A_z @ z / sqrt(s)
        LBz = cp.Parameter((N_AUG, 1))                   # L @ B_z / sqrt(s)
        V_bound = cp.Parameter(nonneg=True)              # (1 - alpha) * V_now / s
        slack_weight = cp.Parameter(nonneg=True)

        Lz_next = LAz_z + LBz @ cp.reshape(u_cbf, (1, 2), order="C")
        constraints += [cp.sum_squares(Lz_next) <= V_bound + delta]
        objective = objective + slack_weight * delta

    prob = cp.Problem(cp.Minimize(objective), constraints)

    def solve(u_r_val, CpApxp_val, CpBp_val, rho_val,
              LAz_z_val=None, LBz_val=None, V_bound_val=None, slack_weight_val=None,
              V_now_val=None):
        u_r.value, CpApxp.value, CpBp.value, rho.value = u_r_val, CpApxp_val, CpBp_val, rho_val
        if use_clf:
            LAz_z.value, LBz.value = LAz_z_val, LBz_val
            V_bound.value, slack_weight.value = V_bound_val, slack_weight_val

        tol = dict(tol_gap_abs=1e-6, tol_gap_rel=1e-6, tol_feas=1e-6)
        used_fallback = False
        try:
            prob.solve(solver=cp.CLARABEL, **tol)
        except cp.error.SolverError:
            used_fallback = True
            try:
                prob.solve(solver=cp.SCS)
            except cp.error.SolverError:
                solve.info = {"status": "solver_error", "delta": np.nan,
                              "V_now": V_now_val, "fallback": True}
                return u_r_val, False

        solve.info = {"status": prob.status,
                      "delta": float(delta.value) if use_clf and delta.value is not None else 0.0,
                      "V_now": V_now_val, "fallback": used_fallback}
        if u_cbf.value is None:
            return u_r_val, False
        return u_cbf.value, True

    solve.info = {}
    return solve


def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, u_prev,
              R=SAFE_SET_RADIUS, gamma=0.1, margin=0.0, safe_tolerance=SAFE_TOLERANCE,
              current_aoi=1,
              z=None, A_z=None, B_z=None, P=None, alpha=0.1, slack_weight=10.0):
    '''
    The cbf works only on the REMOTE side. Use the function 'solve' to perform the cvxpy solve.
    z is the (error-coordinate) augmented state, 5x2 (one column per DOF).
    If z/A_z/B_z/P are None, the CLF part is skipped (requires make(use_clf=False)).
    '''
    CpApxp = (Cp @ Ap @ x_plant).ravel()
    CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-16:
        return np.asarray(u_remote).reshape(1, 2)

    rho = np.sqrt(max((R - margin)**2 - (1 - gamma) * ((R - margin)**2 - np.sum(y_remote**2)), 0))

    if z is None:
        u_cbf, s_ok = solve(u_remote.ravel(), CpApxp, CpBp, rho)
    else:
        P_sym = (P + P.T) / 2
        L = np.linalg.cholesky(P_sym).T                  # P = L^T L
        V_now = float(np.sum((L @ z)**2))                # sum_k z_k^T P z_k
        s = max(V_now, 1)                              # normalization scale
        LAz_z = (L @ (A_z @ z)) / np.sqrt(s)             # 5x2
        LBz = (L @ B_z) / np.sqrt(s)                     # 5x1
        V_bound = (1 - alpha) * V_now / s
        u_cbf, s_ok = solve(u_remote.ravel(), CpApxp, CpBp, rho,
                            LAz_z, LBz, V_bound, slack_weight, V_now)
    # print("solve info:", solve.info)
    return np.asarray(u_cbf).reshape(1, 2)