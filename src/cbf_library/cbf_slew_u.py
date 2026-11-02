
from experiment_go import *

import cvxpy as cp
import numpy as np

def make():
    u = cp.Variable(2); slack = cp.Variable(nonneg=True)
    u_r, u_prev = cp.Parameter(2), cp.Parameter(2)
    y0, a, rho = cp.Parameter(2), cp.Parameter(), cp.Parameter(nonneg=True)
    delta, w = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(u - u_r) + w * cp.square(slack)),
        [cp.norm(y0 + a * u) <= rho,                  # hard: wall
         cp.norm(u - u_prev) <= delta + slack])       # soft: slew limit
    def solve(u_r_v, u_prev_v, y0_v, a_v, rho_v, delta_v, w_v):
        (u_r.value, u_prev.value, y0.value, a.value,
         rho.value, delta.value, w.value) = (u_r_v, u_prev_v, y0_v, a_v, rho_v, delta_v, w_v)
        prob.solve(solver=cp.CLARABEL)
        if u.value is None:
            raise ValueError(f"CBF failed: {prob.status}")
        return u.value
    return solve

def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, current_aoi,
              u_prev, R=SAFE_SET_RADIUS, gamma=0.1, thr=AOI_THRESHOLD,
              d_hi=5e-4, d_lo=1e-6, w=100.0):
    print("u_prev:", u_prev)
    a = (Cp @ Bp).item()
    if abs(a) < 1e-12:                                   # safe mode: bypass
        return u_remote
    s = 1/(1 + np.exp(-(current_aoi - 0.5*thr)))                 # ramps up before the switch
    delta = d_hi*(1 - s) + d_lo*s                        # loose -> tight with age
    rho = np.sqrt(max(R**2 - (1-gamma)*(R**2 - np.sum(y_remote**2)), 0))
    u = solve(u_remote.ravel(), u_prev.ravel(), (Cp @ Ap @ x_plant).ravel(),
              a, rho, delta, w)
    return np.asarray(u).reshape(1, 2)