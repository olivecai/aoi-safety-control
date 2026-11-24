'''
Local-limit CBF (wall disc + AoI-shrinking local disc) with a soft CLF, where the augmented
state keeps a CHOSEN number of past steps:

    z(t) = [ x(t), x(t-1), ..., x(t-n_x),  u(t-1), u(t-2), ..., u(t-n_u) ]

n_x = n_u = 1 reproduces the earlier z = [x(t), x(t-1), u(t-1)].
n_x = 0, n_u = 0 gives the plain x-only state.

QP solved each step (y_next = Cp Ap x + Cp Bp u):
    min_{u, delta}  ||u - u_remote||^2 + slack_weight * delta
    s.t.  ||y_next||        <= rho        wall disc (hard)
          ||y_next - y_t||  <= r(AoI)     local disc (hard)
          V(z_next)/s <= (1 - alpha) V(z)/s + delta,  delta >= 0     (CLF, soft)
Only r depends on AoI. alpha and slack_weight must not.
'''

from experiment_go import *

import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are

DEBUG = False


# ---------------------------------------------------------------- augmented model
def build_augmented(Ap, Bp, n_x=1, n_u=1):
    '''Shift-register model: z+ = A_z z + B_z u, z = [x(t..t-n_x), u(t-1..t-n_u)].'''
    n, m = Ap.shape[0], Bp.shape[1]
    dx, du = n * (n_x + 1), m * n_u
    N = dx + du
    A_z = np.zeros((N, N))
    B_z = np.zeros((N, m))

    A_z[0:n, 0:n] = Ap                                   # x(t+1) = Ap x(t) + Bp u(t)
    B_z[0:n, :] = Bp
    for k in range(1, n_x + 1):                          # x(t-k) at t+1 is x(t-k+1) at t
        A_z[k * n:(k + 1) * n, (k - 1) * n:k * n] = np.eye(n)
    if n_u > 0:
        B_z[dx:dx + m, :] = np.eye(m)                    # u(t) becomes u((t+1)-1)
        for j in range(1, n_u):                          # shift older inputs
            A_z[dx + j * m:dx + (j + 1) * m, dx + (j - 1) * m:dx + j * m] = np.eye(m)
    return A_z, B_z


def clf_params(Ap, Bp, n_x=1, n_u=1, q_vel=1.0, eps_pos=1e-2, stale_scale=1.0,
               eps_u=1e-2, r=1.0):
    '''
    P from LQR on the augmented model (plant-side only, no reference).
    Assumes x = [position, velocity]. Weights: current [eps_pos, q_vel],
    stale x blocks [eps_pos, stale_scale*q_vel], each u(t-k) gets eps_u.
    '''
    n, m = Ap.shape[0], Bp.shape[1]
    A_z, B_z = build_augmented(Ap, Bp, n_x, n_u)
    diag = [eps_pos, q_vel]
    diag += [eps_pos, stale_scale * q_vel] * n_x
    diag += [eps_u] * (m * n_u)
    Q_z = np.diag(diag)
    R = r * np.eye(m)
    P = solve_discrete_are(A_z, B_z, Q_z, R)
    P = (P + P.T) / 2
    K = np.linalg.solve(R + B_z.T @ P @ B_z, B_z.T @ P @ A_z)
    rho_cl = np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K)))
    assert rho_cl < 1, f"K not stabilizing, spectral radius {rho_cl}"
    assert np.all(np.linalg.eigvalsh(P) > 0), "P not positive definite"
    return A_z, B_z, P, K


class History:
    '''Keeps the last n_x states and n_u applied commands and builds z (N x n_dof).'''
    def __init__(self, x0, n_x=1, n_u=1, m=1):
        n_dof = x0.shape[1]
        self.x = [x0.copy() for _ in range(n_x + 1)]            # x(t), x(t-1), ...
        self.u = [np.zeros((m, n_dof)) for _ in range(n_u)]     # u(t-1), u(t-2), ...

    def z(self):
        return np.vstack(self.x + self.u)

    def push(self, x_next, u_applied):
        '''Call once per step AFTER applying u(t): pass x(t+1) and the applied u(t).'''
        self.x = [x_next.copy()] + self.x[:-1]
        if self.u:
            self.u = [np.asarray(u_applied).reshape(self.u[0].shape).copy()] + self.u[:-1]


# ---------------------------------------------------------------- QP
def make(n_aug, use_clf=True):
    u = cp.Variable(2)
    u_r = cp.Parameter(2)
    CpApxp = cp.Parameter(2)
    CpBp = cp.Parameter()
    rho = cp.Parameter(nonneg=True)
    diff = cp.Parameter(2)
    r = cp.Parameter(nonneg=True)

    constraints = [cp.norm(CpApxp + CpBp * u) <= rho,       # wall disc
                   cp.norm(CpBp * u - diff) <= r]           # local disc, == ||y_next - y_t|| <= r
    objective = cp.sum_squares(u - u_r)

    if use_clf:
        delta = cp.Variable(nonneg=True)
        LAz_z = cp.Parameter((n_aug, 2))
        LBz = cp.Parameter((n_aug, 1))
        V_bound = cp.Parameter(nonneg=True)
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
    return r_max / (1 + np.exp(r_slope * (current_aoi - 1)))


def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, u_prev, R=SAFE_SET_RADIUS,
              gamma=0.1, current_aoi=1, aoi_threshold=AOI_THRESHOLD,
              z=None, A_z=None, B_z=None, P=None, alpha=0.05, slack_weight=1.0):
    CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-12:
        return np.asarray(u_remote).reshape(1, 2)

    y_t = np.asarray(y_remote).ravel()
    CpApxp = (Cp @ Ap @ x_plant).ravel()
    diff = y_t - CpApxp
    r = local_radius(current_aoi)
    rho = np.sqrt(max(R**2 - (1 - gamma) * (R**2 - y_t @ y_t), 0))

    if z is None:
        u, ok = solve(u_remote.ravel(), CpApxp, CpBp, rho, diff, r)
    else:
        L = np.linalg.cholesky((P + P.T) / 2).T          # P = L^T L
        V_now = float(np.sum((L @ z) ** 2))
        s = max(V_now, 1.0)
        LAz_z = (L @ (A_z @ z)) / np.sqrt(s)
        LBz = (L @ B_z) / np.sqrt(s)
        V_bound = (1 - alpha) * V_now / s
        u, ok = solve(u_remote.ravel(), CpApxp, CpBp, rho, diff, r,
                      LAz_z, LBz, V_bound, slack_weight, V_now)

    if not ok:
        u = diff / CpBp          # hold the output: inside both discs

    if DEBUG:
        y_next = CpApxp + CpBp * u
        print(f"AoI {current_aoi}, r {r:.4f}, |y_next|={np.linalg.norm(y_next):.4f} (rho {rho:.4f}), "
              f"|y_next-y_t|={np.linalg.norm(y_next - y_t):.4f}, info={getattr(solve, 'info', {})}")
    return np.asarray(u).reshape(1, 2)