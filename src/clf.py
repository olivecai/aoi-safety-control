'''
Get CLF

AUGMENTED SYSTEM:
z(t) = [x(t), x(t-1), u(t-1)]
'''

'''
Need to find K

EXAMPLE:
Suppose u(t) = -K @ x(t)
Then our plant state  x(t+1) = Ax(t)+Bu(t) becomes:
x(t+1) = Ax(t)-BKx(t) = (A-BK) x(t)

FIND K for AUGMENTED SYSTEM:
z(t) = [x(t), x(t-1), u(t-1)]
z(t+1) = Az @ z(t) + Bz @ u(t)

n, m = A.shape[0], B.shape[1]
n == 2
m == 1

A_z = np.block([
    [A,          np.zeros((n, n)), np.zeros((n, m))],
    [np.eye(n),  np.zeros((n, n)), np.zeros((n, m))],
    [np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, m))],
])
B_z = np.vstack([B, np.zeros((n, m)), np.eye(m)])

'''
from experiment_go import *
from building_block import *
import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are



# this builds the augmented Az and Bz matrices for the augemented state z
# and we only need to do this once using the matrices Ap[HUMAN MODE] and Bp[HUMAN MODE] 
def build_augmented(Ap, Bp):
    n, m = Ap.shape[0], Bp.shape[1]
    A_z = np.block([
        [Ap,               np.zeros((n, n)), np.zeros((n, m))],
        [np.eye(n),        np.zeros((n, n)), np.zeros((n, m))],
        [np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, m))],
    ])
    B_z = np.vstack([Bp, np.zeros((n, m)), np.eye(m)])
    return A_z, B_z

def clf_params(Ap, Bp, q_vel=1.0, eps_pos=1e-2, eps_u=1e-2, r=1.0):
    n, m = Ap.shape[0], Bp.shape[1]
    A_z, B_z = build_augmented(Ap, Bp)
    Q_z = np.diag([eps_pos, q_vel,      # current [pos, vel]
                   eps_pos, q_vel,      # stale   [pos, vel]
                   eps_u])              # u(t-1)
    R = r * np.eye(m)
    P = solve_discrete_are(A_z, B_z, Q_z, R)
    P = (P + P.T) / 2
    K = np.linalg.solve(R + B_z.T @ P @ B_z, B_z.T @ P @ A_z)
    rho = np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K)))
    assert rho < 1, f"K not stabilizing, spectral radius {rho}"
    assert np.all(np.linalg.eigvalsh(P) > 0), "P not positive definite"
    return A_z, B_z, P, K

def simulate_run(N_s, h_seq, aoi_threshold_f, aoi_threshold_b, rng, make_cbf, get_u_cbf, switching=True, seed=RNG_SEED, lossless=True, aoi_aware=1):
    '''
    CLF with augmented system z(t) = [x(t) x(t-1) u(t-1)]
    '''
    # # precompute the 8 possible matrices ONCE (avoid rebuild everytime)
    # modes = {(sf, sb, m): build_mode(sf, sb, m)
    #          for sf in (0, 1) for sb in (0, 1) for m in (MODE_HUMAN, MODE_SAFE)}
    # once, before the loop
    clf = {zeta: clf_params(A_plant[zeta], B_plant[zeta]) for zeta in (MODE_HUMAN, MODE_SAFE)}

    if make_cbf is not None:
        solve = make_cbf()
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

    n, m = Ap.shape[0], Bp.shape[1] # in our specific case, n is 2 and m is 1

    # generate the augmented state matrices
    A_z = np.block([
        [Ap,          np.zeros((n, n)), np.zeros((n, m))],
        [np.eye(n),  np.zeros((n, n)), np.zeros((n, m))],
        [np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, m))],
    ])
    B_z = np.vstack([Bp, np.zeros((n, m)), np.eye(m)])


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
    u_prev_eff = np.zeros((1,2))
 
    # z(t) = [x(t), x(t-1), u(t-1)]
    # z(t+1) = Az @ z(t) + Bz @ u(t)

    # z = np.concatenate((x_plant, x_plant_prev, u_prev_eff))
    # print(f"Initial State:\n{x_plant}")

    # DO NOT USE THE LINES BELOW this was when human reference was added in just to practice CLF
    # x_ref = np.zeros((2, 2))
    # x_ref[0, :] = h / Cp[0, 0]
    # x_ref_prev = x_ref.copy()

    x_plant_prev = x_plant.copy()

    for t in range(N_s):
        z = np.vstack([x_plant, x_plant_prev, u_prev_eff])   # 5x2
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

        # not directly needed for augmented state :
        Cp = C_plant[zeta]
        Ap = A_plant[zeta]
        Bp = B_plant[zeta]

        A_z, B_z, P, K = clf[zeta]
        print("eig P:", np.linalg.eigvalsh(P))
        print("cond P:", np.linalg.cond(P))
        # z = np.vstack([x_plant, x_plant_prev, u_prev_eff])  #5x2, one column per dof

        Cc = C_controller[zeta]
        Ac = A_controller[zeta]
        Bc = B_controller[zeta]
        Dc = D_controller[zeta]

        h = h_seq[:,t]

        # x_ref_prev = x_ref.copy()                 # last step's reference
        # x_ref = np.zeros((2, 2))
        # x_ref[0, :] = h / Cp[0, 0]                # this step's reference (zero velocity)
        # u_ref = np.zeros((1, 2))
        # z_err = np.vstack([x_plant - x_ref, x_plant_prev - x_ref_prev, u_prev_eff - u_ref])
        # print("pos/vel err :", np.linalg.norm(x_plant - x_ref))
        # print("stale err   :", np.linalg.norm(x_plant_prev - x_ref_prev))
        # print("u block     :", np.linalg.norm(u_prev_eff - u_ref))
        # current_aoi = max(aoi_b, aoi_f) if aoi_aware else 1
        
    

        current_aoi = aoi_f if aoi_aware else 1

      
        # u_ref = np.zeros((1, 2))                         # steady-state command; 0 for a double integrator
        # z_err = np.vstack([x_plant - x_ref, x_plant_prev - x_ref_prev, u_prev_eff - u_ref])
        # print("V_now", solve.info.get("V_now"), "||z_err||", np.linalg.norm(z_err))

        if get_u_cbf is not None:
            u_cbf = get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote,
                  current_aoi=current_aoi, u_prev=u_prev_eff,
                  z=z, A_z=A_z, B_z=B_z, P=P)   # no AoI-dependence in the CLF part            
            x_plant_cbf = Ap @ x_plant + Bp @ u_cbf #safe
            u_prev_cbf = u_cbf

        #non cbf version is calculated either way
        x_plant_ncbf = Ap @ x_plant + Bp @ u_remote #safe
        u_prev_ncbf=u_remote

        x_plant_prev = x_plant.copy()

        if get_u_cbf is not None:
            x_plant, u_prev_eff = x_plant_cbf , u_prev_cbf
        else:
            x_plant, u_prev_eff = x_plant_ncbf , u_prev_ncbf

        

        y_remote = Cp @ x_plant
        y_local = y_remote if sigma_b else y_local #update y_local (local feedback controller side) if the backwards link worked

        # uncomment and untab to get this back
                # # add the y_local cbf here to get y_aoi_aware := assume continually decellerating drift in the direction of the last command given (more helpfully would also be the curl of the command given).
                # m_pos = 0.5
                # bd_lower, bd_upper = 0, 1
                # y_aoi_aware_t_parameter = bd_lower + (bd_upper-bd_lower)/(1+np.e**(-m_pos*(aoi_b-3)))
                # # y_assumption 
                        
                # y_assumption = (Cp @ x_plant_ncbf)
                
                # y_aoi_aware = (1-y_aoi_aware_t_parameter) *y_local + y_aoi_aware_t_parameter * y_assumption  
                
                # print("LOG y_local", y_local, "y_remote", y_remote)
                # print("LOG difference between y local and remote:", np.linalg.norm([y_local- y_remote]))
                # print(f"LOG aoi_f: {aoi_f}, aoi_b: {aoi_b}")
                # # print("h", h)
                # # print(h-y_local)
                # y_local  = y_assumption
        tracked_y_remote[:,t] = y_remote.ravel()

        x_controller = Ac @ x_controller + Bc @ (h - y_local)
        u_local = Cc @ x_controller + Dc @ ( h - y_local)
        u_remote = u_local if sigma_f else u_remote # update u_remote (remote command on plant side) if the forwards link worked
        print("LOG u_local", u_local, "u_remote", u_remote)
        print("LOG difference between u local and remote:", np.linalg.norm([u_local- u_remote]))
        print(f"LOG aoi_f: {aoi_f}, aoi_b: {aoi_b}")
     

    return tracked_y_remote