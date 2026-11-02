
from experiment_go import *

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
def make():
    u = cp.Variable(2); 
    u_r = cp.Parameter(2)
    CpApxp, CpBp, rho = cp.Parameter(2), cp.Parameter(), cp.Parameter(nonneg=True)
    difference_between_yt_CpApxp, r, w = cp.Parameter(2), cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(u - u_r)),
        [cp.norm(CpApxp + CpBp*u) <= rho,              # hard: wall disc, centred at 0
         cp.norm(CpBp*u - difference_between_yt_CpApxp) <= r ])        # soft: local disc, centred at y_t
    def solve(u_r_v, CpApxp_v, CpBp_v, rho_v, difference_between_yt_CpApxp_v, r_v, w_v):
        (u_r.value, CpApxp.value, CpBp.value, rho.value,
         difference_between_yt_CpApxp.value, r.value, w.value) = u_r_v, CpApxp_v, CpBp_v, rho_v, difference_between_yt_CpApxp_v, r_v, w_v
        prob.solve(solver=cp.CLARABEL, verbose=True)
        if u.value is None:
            raise ValueError(f"CBF failed: {prob.status}")
        return u.value, 0
    return solve

def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, u_prev, R=SAFE_SET_RADIUS,
              gamma=0.1, current_aoi=1, aoi_threshold=AOI_THRESHOLD,
              r_hi=1e-2, r_lo=0, kappa=100.0):
    CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-12:
        return u_remote
    y_t = np.asarray(y_remote).ravel() #current y output (both position and vecloity)
    s = 1/(1 + np.exp(-(current_aoi - 0.5*aoi_threshold)))
    # r_max = np.linalg.norm(R-np.linalg.norm(y_t))# distance from y_t to the boundary of the local disc, which is a function of AOI
    r_max=1
    r_slope = 1.0
    bd_lower, bd_upper = 0, r_max
    r = bd_lower + (bd_upper-bd_lower)/(1+np.e**(r_slope*(current_aoi-1)))

    # plot r:
    # plt.plot(np.linspace(0, 30, 100),bd_lower + (bd_upper-bd_lower)/(1+np.e**(r_slope*(np.linspace(0, 30, 100)-(aoi_threshold/6)))))
    # plt.show()
    # exit()
    print(f"r: {r}, aoi: {current_aoi}")
    CpApxp = (Cp @ Ap @ x_plant).ravel()

    m_pos = 0.1
    bd_lower, bd_upper = 0, gamma
    cbf_shift = AOI_THRESHOLD
    gamma_aoi = bd_lower + (bd_upper-bd_lower)/(1+np.e**(m_pos*(current_aoi-cbf_shift)))

    rho = np.sqrt(max(R**2 - (1-gamma)*(R**2 - y_t @ y_t), 0))
    u, slack = solve(u_remote.ravel(), CpApxp, CpBp, rho, y_t - CpApxp, r, kappa/CpBp**2)
    print(f"cp.norm(CpApxp + CpBp*u):{np.linalg.norm(CpApxp + CpBp*u)}, rho: {rho}")
    print(f"np.norm(CpBp*u - difference_between_yt_CpApxp) ", np.linalg.norm(CpBp*u - (y_t - CpApxp)), "r",r)
    print(f"y_t - CpApxp: {y_t - CpApxp},\n u: {u}, slack:{slack},\n y_next: {CpApxp+(CpBp*u)}")
    return np.asarray(u).reshape(1, 2)