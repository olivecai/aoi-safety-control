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
    u_cbf = cp.Variable(2)
    slack = cp.Variable(nonneg=True) 
    u_r = cp.Parameter(2) 
    CpApxp, CpBp, rho = cp.Parameter(2), cp.Parameter(), cp.Parameter(nonneg=True) 
    pos_Apxp, pos_Bp, rho_pos = cp.Parameter(2), cp.Parameter(), cp.Parameter(nonneg=True) 
    
    prob = cp.Problem( 
        cp.Minimize(cp.sum_squares(u_cbf - u_r) + 1e4*slack), 
        [
            cp.norm(CpApxp + CpBp*u_cbf) <= rho,                # original: hard 
            cp.norm((pos_Apxp + pos_Bp*u_cbf)-(pos_Apxp + pos_Bp*u_r)) <= rho_pos + slack # velocity: soft
        ]
    ) 

    def solve(u_r_val, CpApxp_val, CpBp_val, rho_val, pos_Apxp_val, pos_Bp_val, rho_pos_val): 
        u_r.value, CpApxp.value, CpBp.value, rho.value = u_r_val, CpApxp_val, CpBp_val, rho_val 
        pos_Apxp.value, pos_Bp.value, rho_pos.value = pos_Apxp_val, pos_Bp_val, rho_pos_val 
        # print(pos_Apxp.value, pos_Bp.value, rho_pos.value)
        # print("v")
        # print(np.linalg.norm(pos_Apxp.value + pos_Bp.value*u_r.value))
        
        prob.solve(solver=cp.CLARABEL, verbose=False) 
        
        # Check if the optimization problem failed to find a feasible solution
        if prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            raise ValueError(f"CBF Optimization failed: Problem is {prob.status}")
            
        # Catch-all for other failure modes (e.g., UNBOUNDED, SOLVER_ERROR)
        if u_cbf.value is None:
            raise ValueError(f"Optimization failed with status: {prob.status}")


        return u_cbf.value, slack.value

    return solve



def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, R=SAFE_SET_RADIUS, gamma=0.1, margin=0.0, safe_tolerance=SAFE_TOLERANCE, current_aoi=1):

    m_pos = 0.1
    bd_lower, bd_upper = 0, gamma/5
    position_cbf_shift = AOI_THRESHOLD
    gamma_position = bd_lower + (bd_upper-bd_lower)/(1+np.e**(m_pos*(current_aoi-position_cbf_shift)))

    m = 1
    gamma_std = bd_lower + (bd_upper-bd_lower)/(1+np.e**(m*(current_aoi-AOI_THRESHOLD)))

    P=0.1
    s = 1/(1+np.exp(-1e-2*(current_aoi - AOI_THRESHOLD)))  
    k=1.0
    P = P * (1 - k*s)
    CpApxp = (Cp @ Ap @ x_plant).ravel(); CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-16: return u_remote        

    Cp_vel = np.array([Cp.ravel()[0], 0.0])
    pos_Apxp_val = (Cp_vel @ Ap @ x_plant).ravel(); pos_Bp_val = (Cp_vel @ Bp).item()
    y_pos_now = Cp_vel @ x_plant
    print(y_pos_now)
    print((P)**2)
    print(np.sum(y_pos_now**2))
    print("---")
    

    rho_val     = np.sqrt(max(R**2 - (1-gamma_std)*(R**2 - np.sum(y_remote**2)), 0))
    rho_pos_val = np.sqrt(max(P**2 - (1-gamma_position)*((P)**2 - np.sum(y_pos_now**2)), 0))
    
    u_r_val = u_remote.ravel()
    
    u_cbf, s = solve(u_r_val, CpApxp, CpBp, rho_val, pos_Apxp_val, pos_Bp_val, rho_pos_val) #, pos_Apxp_val, pos_Bp_val, rho_pos_val)
    # print("aoi, solved u_cbf, s:", current_aoi, u_cbf, s)


    vel_next = np.linalg.norm(pos_Apxp_val + pos_Bp_val * u_cbf.ravel())

    return np.asarray(u_cbf).reshape(1,2)