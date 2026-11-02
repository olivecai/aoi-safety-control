'''
november 1 2026
problem: 
1. the original cbf (see `standard.py`) was very effective at keeping position inside safe set, but allowed for high velocity oscillations inside the cbf itself when far away from the cbf boundary.
2. my first instinct to address this problem was add an extra penalization on velocity.
3. however, this made the system (obviuosly) very very slow and the human desired reference changed too quickly for the system and resulted in a lot of lag.
4. but upon further reflection, what i want to retain is the velocity and what i want to control is the position of the system; because if the system has a very accurate sense of where it should go, then the velocity it needs to get there will naturally be more reasonable.
'''

from experiment_go import *

import cvxpy as cp
import numpy as np
def make():
    u = cp.Variable(2); slack = cp.Variable(nonneg=True)
    u_r = cp.Parameter(2)
    y0, a, rho = cp.Parameter(2), cp.Parameter(), cp.Parameter(nonneg=True)
    p0, b_p, rho_p = cp.Parameter(2), cp.Parameter(), cp.Parameter(nonneg=True)
    w = cp.Parameter(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(u - u_r) + w * cp.square(slack)),
        [cp.norm(y0 + a*u) <= rho,                  # hard: original wall
         cp.norm(p0 + b_p*u) <= rho_p + 1e2*slack])     # soft: AoI-dependent position disc
    def solve(u_r_v, y0_v, a_v, rho_v, p0_v, b_p_v, rho_p_v, w_v):
        (u_r.value, y0.value, a.value, rho.value,
         p0.value, b_p.value, rho_p.value, w.value) = (u_r_v, y0_v, a_v, rho_v,
                                                       p0_v, b_p_v, rho_p_v, w_v)
        prob.solve(solver=cp.CLARABEL)
        if u.value is None:
            raise ValueError(f"CBF failed: {prob.status}")
        return u.value, slack.value
    return solve

def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, R=SAFE_SET_RADIUS,
              gamma=0.1, current_aoi=1, thr=AOI_THRESHOLD, P_min=0.1, w=1e3):
    a = (Cp @ Bp).item()
    if abs(a) < 1e-12:
        return u_remote
    Ep = np.array([Cp.ravel()[0], 0.0])              # position part of y is the (15*p)
    s = 1/(1 + np.exp(-(current_aoi - 0.5*thr)))     
    P = R - (R - P_min)*s                            # R at low AoI, P_min at high AoI ???
    y_pos = Ep @ x_plant
    rho   = np.sqrt(max(R**2 - (1-gamma)*(R**2 - np.sum(y_remote**2)), 0))
    rho_p = np.sqrt(gamma*P**2 + (1-gamma)*np.sum(y_pos**2))
    u, _ = solve(u_remote.ravel(), (Cp @ Ap @ x_plant).ravel(), a, rho,
                 (Ep @ Ap @ x_plant).ravel(), (Ep @ Bp).item(), rho_p, w)
    return np.asarray(u).reshape(1, 2)