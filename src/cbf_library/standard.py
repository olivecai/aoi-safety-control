'''
Standard CBF, is NOT AOI AWARE, and weights the output state Y as regular norm instead of extra penalization on velocity or position
'''

from experiment_go import *

import cvxpy as cp
import numpy as np

def make():
    '''
    ret cbf fn
    '''
    u_cbf   = cp.Variable(2)                    # [u_x, u_y]
    u_r = cp.Parameter(2)                   # u_remote, the received command
    CpApxp  = cp.Parameter(2)                   # C_p A_p x for each axis
    CpBp   = cp.Parameter()                    # C_p B_p (scalar)
    rho = cp.Parameter(nonneg=True)

    prob = cp.Problem(objective= cp.Minimize(cp.sum_squares(u_cbf - u_r)),
                      constraints= [cp.norm(CpApxp + CpBp * u_cbf) <= rho])

    def solve(u_r_val, CpApxp_val, CpBp_val, rho_val):
        u_r.value, CpApxp.value, CpBp.value, rho.value = u_r_val, CpApxp_val, CpBp_val, rho_val
        prob.solve(solver=cp.CLARABEL)     
        if u_cbf.value is None:            
            return u_r_val, False
        return u_cbf.value, True
    return solve

def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, R=SAFE_SET_RADIUS, gamma=0.1, margin=0.0, safe_tolerance=SAFE_TOLERANCE, current_aoi=1):

    '''
    the cbf works only on the remote side. use the function 'solve' to perform the cvxpy solve fn
    '''
    CpApxp = (Cp @Ap@x_plant).ravel()
    CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-16:
        return u_remote

    rho=np.sqrt(max((R-margin)**2 - (1-gamma)*((R-margin)**2 - np.sum(y_remote**2)),0))
    u_cbf, s = solve(u_remote.ravel(), CpApxp, CpBp, rho)
    # print("solved u_cbf, s:", u_cbf, s)
    return np.asarray(u_cbf).reshape(1,2)