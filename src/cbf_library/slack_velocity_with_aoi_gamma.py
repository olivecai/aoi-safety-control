'''
This CBF is not only AOI AWARE but has extra penalization on velocity, by using a SLACK constraint!
'''

from experiment_go import *

import cvxpy as cp
import numpy as np

def make(): 
    u_cbf = cp.Variable(2)
    slack = cp.Variable(nonneg=True) 
    u_r = cp.Parameter(2) 
    CpApxp, CpBp, rho = cp.Parameter(2), cp.Parameter(), cp.Parameter(nonneg=True) 
    vel_Apxp, vel_Bp, rho_vel = cp.Parameter(2), cp.Parameter(), cp.Parameter(nonneg=True) 
    
    prob = cp.Problem( 
        cp.Minimize(cp.sum_squares(u_cbf - u_r) + 1e4*slack), 
        [
            cp.norm(CpApxp + CpBp*u_cbf) <= rho,                # original: hard 
            cp.norm(vel_Apxp + vel_Bp*u_cbf) <= rho_vel + slack # velocity: soft
        ]
    ) 

    def solve(u_r_val, CpApxp_val, CpBp_val, rho_val, vel_Apxp_val, vel_Bp_val, rho_vel_val): 
        u_r.value, CpApxp.value, CpBp.value, rho.value = u_r_val, CpApxp_val, CpBp_val, rho_val 
        vel_Apxp.value, vel_Bp.value, rho_vel.value = vel_Apxp_val, vel_Bp_val, rho_vel_val 
        # print(vel_Apxp.value, vel_Bp.value, rho_vel.value)
        print("v")
        print(np.linalg.norm(vel_Apxp.value + vel_Bp.value*u_r.value))
        
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
    '''
    for each DOF, split position from velocity: 
        Y = [[position_x, velocity_x], # dof 1
            [position_y, velocity_y],  # dof 2
            [position_z, velocity_z]] # dof 3
    Y_pos = [[position_x, position_y, position_z]]
    Y_vel = [[velocity_x, velocity_y, velocity_z]]

    # cbf should focus on position same throughout, 
    # but should focus on velocity (more than?) position as aoi increases
    # QUESTION should velocity and position be seperate or inherently linked...

    when we introduce a parameter, we have to have rule:
    - if current aoi goes to 1 every time, then the cbf must behave just like the 
    '''

    # Cp_pos = np.array([Cp.ravel()[0], 0.0])
    # Cp_vel = np.array([0.0, Cp.ravel()[1]])
    # assert np.allclose(Cp_pos + Cp_vel, Cp.ravel())

    # vel_Apxp_val = (Cp_vel @ Ap @ x_plant).ravel()
    # vel_Bp_val   = (Cp_vel @ Bp).item()
    # pos_Apxp_val = (Cp_pos @ Ap @ x_plant).ravel()
    # pos_Bp_val   = (Cp_pos @ Bp).item()

    # # print(CpApxp.shape)
    # # print(pos_Apxp_val.shape)
    # print("@@@")
    # print(pos_Bp_val)
    # print(vel_Bp_val)
    # print(Cp@Bp)
    # dangerous = np.min((pos_Apxp_val,vel_Apxp_val))
    # if abs(dangerous) < 1e-16:
    #     return u_remote
  
    # gamma now a function of current_aoi, everything else is just a const:
    m_vel = 0.1 #yooo later we can make this a function too?!?!?!? maybe not tractable tho
    bd_lower, bd_upper = 0, gamma/5
    velocity_cbf_shift = AOI_THRESHOLD
    gamma_velocity = bd_lower + (bd_upper-bd_lower)/(1+np.e**(m_vel*(current_aoi-velocity_cbf_shift)))

    # # we want to see the gammaa function
    # plot_x = np.linspace(0,100,200)
    # plot_y = np.zeros(plot_x.shape)
    # for i in range(len(plot_x)):
    #     current_aoi = plot_x[i]
    #     gamma_velocity = bd_lower + (bd_upper-bd_lower)/(1+np.e**(m_vel*(current_aoi-velocity_cbf_shift)))
    #     plot_y[i] = gamma_velocity
        
    # plt.plot(plot_x,plot_y)
    # plt.show()

    # print("gamma_velocity", gamma_velocity)
    # rho_vel_val=np.sqrt(max((R-margin)**2 - (1-gamma_velocity)*((R-margin)**2 - np.sum(y_remote**2)),0))


    m_pos = 1
    gamma_position = bd_lower + (bd_upper-bd_lower)/(1+np.e**(m_pos*(current_aoi-AOI_THRESHOLD)))
    # print("gamma_position", gamma_position)
    # gamma_velocity=0.1
    # gamma_position=0.1

    # rho_pos_val = np.sqrt(max((R-margin)**2 - (1-gamma_position)*((R-margin)**2 - np.sum(y_remote**2)),0))

    # gamma = ((np.arctan(m(current_aoi-AOI_THRESHOLD)))/np.pi) + .5
    V=0.01
    s = 1/(1+np.exp(-1e-4*(current_aoi - AOI_THRESHOLD)))   # 0 at low AOI, 1 at high AOI
    k=1.0
    # R = R * (1 - k*s)                                  # e.g. k = 0.3
    V = V * (1 - k*s)
    # print("R,V:", R,V)
    CpApxp = (Cp @ Ap @ x_plant).ravel(); CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-16: return u_remote          # restore the original guard

    Cp_vel = np.array([0.0, Cp.ravel()[1]])
    vel_Apxp_val = (Cp_vel @ Ap @ x_plant).ravel(); vel_Bp_val = (Cp_vel @ Bp).item()
    y_vel_now = Cp_vel @ x_plant

    rho_val     = np.sqrt(max(R**2 - (1-gamma_position)*(R**2 - np.sum(y_remote**2)), 0))
    rho_vel_val = np.sqrt(max(V**2 - (1-gamma_velocity)*(V**2 - np.sum(y_vel_now**2)), 0))
    
    u_r_val = u_remote.ravel()
    
    u_cbf, s = solve(u_r_val, CpApxp, CpBp, rho_val, vel_Apxp_val, vel_Bp_val, rho_vel_val) #, pos_Apxp_val, pos_Bp_val, rho_pos_val)
    # print("aoi, solved u_cbf, s:", current_aoi, u_cbf, s)


    vel_next = np.linalg.norm(vel_Apxp_val + vel_Bp_val * u_cbf.ravel())

    # print(f"aoi={current_aoi} vel_next={vel_next:.4f} rho_vel={rho_vel_val:.4f} "f"|u_cbf-u_r|={np.linalg.norm(u_cbf.ravel()-u_r_val):.4f}")


    return np.asarray(u_cbf).reshape(1,2)