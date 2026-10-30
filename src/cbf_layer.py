'''
This file is invoked by main.py as the behind the scenes logic of our simualtion; this file contains:
- the quadratic programming / optimization solvign with CVXPY to solve our CBFs via import of the cbf from `src/cbf_library`
- generation of our safe set.

This file invokes functions from building_blocks.py and imports variables from experiment_go.py.
'''
from experiment_go import *
from building_block import *
import cvxpy as cp
import numpy as np

def simulate_run_cbf(N_s, h_seq, aoi_threshold_f, aoi_threshold_b, rng, make_cbf, get_u_cbf, switching=True, seed=RNG_SEED, lossless=True):
    '''
    pass in a trajecotry h_seq for 2 dof system 
    set switching and cbf_mode

    then run simulation and get Y, the actual output trajectory
    '''
    # # precompute the 8 possible matrices ONCE (avoid rebuild everytime)
    # modes = {(sf, sb, m): build_mode(sf, sb, m)
    #          for sf in (0, 1) for sb in (0, 1) for m in (MODE_HUMAN, MODE_SAFE)}
    if make_cbf is not None:
        solve = make_cbf()
    z = np.zeros(8) #empty augmented state z(t) = [Xp(t)_T Xc(t)_T Ur(t-1) Yl(t-1)]_T
    sigma_f, sigma_b = 1, 1          # start with both links working
    aoi_f, aoi_b = 1, 1
    tracked_y_remote = np.zeros((2, N_s)) #for position in x and y

    ##### INITIAL STATE #####
    zeta=MODE_HUMAN
    Cp = C_plant[zeta]
    Ap = A_plant[zeta]
    Bp = B_plant[zeta]
    Cc = C_controller[zeta]
    Ac = A_controller[zeta]
    Bc = B_controller[zeta]
    Dc = D_controller[zeta]

    x_controller = np.zeros((4, 2))#now we have two DOFs
    x_plant = np.zeros((2, 2))  # stationary at zero (zero velocity, zero position)
    
    h = h_seq[:,0]
    # all this does, is start the initial position at the proper start point, but it keep velocity 0
    x_plant[0, :] = h / Cp[0, 0]    # y = 15 p  =>  p = h/15
    # print("h:" , h)
    # print(x_plant)
    # exit()
    # print(h.shape)
    y_remote = Cp @ x_plant
    y_local = y_remote # because in the beginning the aoi is 1
    u_local = Cc @ x_controller + Dc @ ( h - y_local)
    u_remote = u_local # because in the beginning aoi is 1
 

    # print(f"Initial State:\n{x_plant}")


    for t in range(N_s):
        # 1. channel, eq (4): bursty Markov success/failure
        if lossless:
            sigma_f = sigma_b = 1
        else:
            sigma_f = int(rng.random() < (p1_f if sigma_f else 1 - p0_f))
            sigma_b = int(rng.random() < (p1_b if sigma_b else 1 - p0_b))

  
        aoi_f = aoi(sigma_f, aoi_f, AOI_CAP)
        aoi_b = aoi(sigma_b, aoi_b, AOI_CAP)
    

        # 3-4. theta and mode, eqs (10)-(11)
        theta = get_theta(aoi_f, aoi_b, aoi_threshold_f, aoi_threshold_b)
        sf, sb, zeta = f_map(theta, aoi_threshold_f, aoi_threshold_b)
        if not switching:
            zeta = MODE_HUMAN        # baseline: never switch to safe mode

        # 5. state update, eq (10)
        # A, B = modes[(sf, sb, zeta)]
        # y_x[t] = Cz @ z # position and velocity of the robot 
        # y_y[t] = Cz @ z # position and velocity of the robot 

        Cp = C_plant[zeta]
        Ap = A_plant[zeta]
        Bp = B_plant[zeta]

        Cc = C_controller[zeta]
        Ac = A_controller[zeta]
        Bc = B_controller[zeta]
        Dc = D_controller[zeta]

        h = h_seq[:,t]

        if get_u_cbf is not None:
            u_cbf = get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, current_aoi=max(aoi_b, aoi_f))
            x_plant = Ap @ x_plant + Bp @ u_cbf #safe
        else:
            x_plant = Ap @ x_plant + Bp @ u_remote #safe
          
        y_remote = Cp @ x_plant
        y_local = y_remote if sigma_b else y_local #update y_local (local feedback controller side) if the backwards link worked
        
        # print("y_local", y_local)
        # print("h", h)
        # print(h-y_local)

        tracked_y_remote[:,t] = y_remote.ravel()

        x_controller = Ac @ x_controller + Bc @ (h - y_local)
        u_local = Cc @ x_controller + Dc @ ( h - y_local)
        u_remote = u_local if sigma_f else u_remote # update u_remote (remote command on plant side) if the forwards link worked
     

    return tracked_y_remote