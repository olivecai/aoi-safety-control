'''
to what extent does AOI help for the CBF?

we will build a CBF based on AOI in our system which experiences delays.
in both cases the cbf is able to act immediately and is on the remote plant side.
the only difference is that in CBF_AOI, the safe set becomes smaller when AOI increases.

suppose the CBF regulates position and velocity.

In our simulation, have AOI maximum age be around 16 to ensure it doesnt completely go off the rails

System matrices

in CBF_BASE 
in CBF_AOI 

two degree of freedom, x and y
live in x-y plane: we are going to start off the system at position 0,0,
and we claim the safe set is a convex polygon or disc (supppose just the unit circle D with radius 1, center 0)
then the command (our human desired intention) h is the parameterized curve (rcost, rsint) and t loops through [0,2pi] and r through (0,1) every ms with a speed h_v (human desire velocity). what we get is a continuously moving spiral.

generate the state of x_x and x_y:
x_x = [p_x, v_x]
x_y = [p_y, y_y]

X = [ x_x, x_y ]T
u = [u_x, u_y]T
A = [[A_p, 0],
     [0,A_p]]
B = [[B_p, 0],
     [0,B_p]]
y = [C_p x_x, C_p x_y]T

safe set is R^2 - ||y||^2, R=1
    

CBF_BASE:
u_cbf = mininize over safe u_cbf in D ||u_cbf-u||^2 such that cbf(u_cbf) > 0   

y_(t+1) (u) = [ C_p A_p x_x , C_p A_p x_y ]T + (C_p B_p) u

||y_0 + a u||^2 <= R^2 - (1- gamma)(R^2 - ||y_t||^2) 

'''
import cvxpy as cp

from aoi_based_switching_control_for_safe_haptic_teleoperation_over_a_wireless_network import *
import numpy as np
from math import cos, sin

import matplotlib.pyplot as plt

SAFE_SET_RADIUS=1.0
SAFE_SET_CENTER= 0 
SAFE_SET_EPSILON = 1e-2 
HUMAN_DESIRED_TRAJ_RADIUS = 1.5
SAFE_TOLERANCE = 1e-10
N_s = 4000
RNG_SEED = 888
AOI_THRESHOLD = 10

USE_NO_CBF=0
USE_STANDARD_CBF=1
USE_AOI_CBF=2

def generate_human_reference(N=N_s, radius=HUMAN_DESIRED_TRAJ_RADIUS, distance_from_boundary=SAFE_SET_EPSILON, turns =1):
    '''
    human ref = parameterized curve in R^2, < r(t) cos(t), r(t) sin(t) >
    
    '''
    r = np.linspace(0, radius - distance_from_boundary, N)
    # r = np.array([HUMAN_DESIRED_TRAJ_RADIUS]*N_s)
    t = np.linspace(0, 2 * np.pi * turns, N)
    return np.array([r * np.cos(t), r * np.sin(t)])

def make_unit_circle_cbf():
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


def get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, R=SAFE_SET_RADIUS, gamma=0.1, margin=0.0, safe_tolerance=SAFE_TOLERANCE):

    '''
    the cbf works only on the remote side. use the function 'solve' to perform the cvxpy solve fn
    '''
    CpApxp = (Cp @Ap@x_plant).ravel()
    CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-16:
        return u_remote
    rho=np.sqrt(max((R-margin)**2 - (1-gamma)*((R-margin)**2 - np.sum(y_remote**2)),0))
    u_cbf, s = solve(u_remote.ravel(), CpApxp, CpBp, rho)
    print("solved u_cbf, s:", u_cbf, s)
    return np.asarray(u_cbf).reshape(1,2)

def make_unit_circle_cbf_with_velocity_penalization_for_high_aoi():
    '''
    idea on making the CBF AOI dependent: when the aoi is higher, subject the velocity term of the x status to a much higher penalization.
    
    in the paper, they switch from human to safe mode upon an AOI threshold (POINT OF INTEREST 1) and they also regulate velocity in safe mode instead of position (POINT OF INTEREST 2)
    POINT OF INTEREST 1: instead of a discrete switch, what about a smoooooth switch; or at least a slightly smooth switch (think sigmoid, arctan)
    POINT OF INTEREST 2: the CBF in git commit 'stable: switching and cbf' is position based. but obviously in the paper and as seen by our results, the cbf should be more velocity based when aoi increases. 

    Take our two points of interest and combine them into a cbf: 
    - let gamma term in CBF be a function of AOI, so that gamma is closer to 1 when the threshold has been reached, but closer to 0 when the aoi is still low
    - optimize velocity and position with different penalization, and increase penalization for velocity when aoi is higher.

    ret cbf fn
    '''
    u_cbf   = cp.Variable(2)                    # [u_x, u_y]
    u_r = cp.Parameter(2)                   # u_remote, the received command

    # get just the velocity
    vel_Apxp  = cp.Parameter(2)                   # [0 1] A_p x for each axis
    vel_Bp   = cp.Parameter()                    # [0 1 ] B_p (scalar)

    # get just the position
    pos_Apxp  = cp.Parameter(2)                   # [1 0] A_p x for each axis
    pos_Bp   = cp.Parameter()                    # [1 0] B_p (scalar)

    CpApxp  = cp.Parameter(2)                   # Cp A_p x for each axis
    CpBp   = cp.Parameter()                    # Cp B_p (scalar)
    
    rho_vel = cp.Parameter(nonneg=True)
    rho_pos = cp.Parameter(nonneg=True)
    # rho = cp.Parameter(nonneg=True)


    prob = cp.Problem(objective= cp.Minimize(cp.sum_squares(u_cbf - u_r)),
                      constraints= [cp.norm((vel_Apxp + vel_Bp * u_cbf)) <= rho_vel,
                                   cp.norm((pos_Apxp + pos_Bp * u_cbf)) <= rho_pos
                                #    cp.norm((CpApxp + CpBp*u_cbf)) <=rho                                   
                                   ])

    def solve(u_r_val, vel_Apxp_val, vel_Bp_val, rho_vel_val, pos_Apxp_val, pos_Bp_val, rho_pos_val):
        u_r.value, vel_Apxp.value, vel_Bp.value, rho_vel.value, pos_Apxp.value, pos_Bp.value, rho_pos.value= u_r_val, vel_Apxp_val, vel_Bp_val, rho_vel_val, pos_Apxp_val, pos_Bp_val, rho_pos_val
        prob.solve(solver=cp.CLARABEL)     
        if u_cbf.value is None:            
            return u_r_val, False
        return u_cbf.value, True
    return solve

def make_cbf_aoi(): 
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
        
        prob.solve(solver=cp.CLARABEL, verbose=True) 
        
        # Check if the optimization problem failed to find a feasible solution
        if prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            raise ValueError(f"CBF Optimization failed: Problem is {prob.status}")
            
        # Catch-all for other failure modes (e.g., UNBOUNDED, SOLVER_ERROR)
        if u_cbf.value is None:
            raise ValueError(f"Optimization failed with status: {prob.status}")


        return u_cbf.value, slack.value

    return solve

def get_u_cbf_aoi(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, R=SAFE_SET_RADIUS, gamma=0.1, margin=0.0, safe_tolerance=SAFE_TOLERANCE, current_aoi=1):
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
    m_vel = 1 #yooo later we can make this a function too?!?!?!? maybe not tractable tho
    bd_lower, bd_upper = 0, gamma
    velocity_cbf_shift = AOI_THRESHOLD
    gamma_velocity = bd_lower + (bd_upper-bd_lower)/(1+np.e**(m_vel*(current_aoi-velocity_cbf_shift)))

    # we want to see the gammaa function
    # plot_x = np.linspace(0,30,50)
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
    s = 1/(1+np.exp(-0.1*(current_aoi - AOI_THRESHOLD)))   # 0 at low AOI, 1 at high AOI
    k=0.3
    # R = R * (1 - k*s)                                  # e.g. k = 0.3
    # V = V * (1 - k*s)
    print("R,V:", R,V)
    CpApxp = (Cp @ Ap @ x_plant).ravel(); CpBp = (Cp @ Bp).item()
    if abs(CpBp) < 1e-16: return u_remote          # restore the original guard

    Cp_vel = np.array([0.0, Cp.ravel()[1]])
    vel_Apxp_val = (Cp_vel @ Ap @ x_plant).ravel(); vel_Bp_val = (Cp_vel @ Bp).item()
    y_vel_now = Cp_vel @ x_plant

    rho_val     = np.sqrt(max(R**2 - (1-gamma_position)*(R**2 - np.sum(y_remote**2)), 0))
    rho_vel_val = np.sqrt(max(V**2 - (1-gamma_velocity)*(V**2 - np.sum(y_vel_now**2)), 0))
    
    u_r_val = u_remote.ravel()
    
    u_cbf, s = solve(u_r_val, CpApxp, CpBp, rho_val, vel_Apxp_val, vel_Bp_val, rho_vel_val) #, pos_Apxp_val, pos_Bp_val, rho_pos_val)
    print("aoi, solved u_cbf, s:", current_aoi, u_cbf, s)


    vel_next = np.linalg.norm(vel_Apxp_val + vel_Bp_val * u_cbf.ravel())
    print(f"aoi={current_aoi} vel_next={vel_next:.4f} rho_vel={rho_vel_val:.4f} "f"|u_cbf-u_r|={np.linalg.norm(u_cbf.ravel()-u_r_val):.4f}")


    return np.asarray(u_cbf).reshape(1,2)
    


def simulate_run_cbf(N_s, h_seq, aoi_threshold_f, aoi_threshold_b, rng, switching=True, cbf_mode=True, cbf=make_unit_circle_cbf_with_velocity_penalization_for_high_aoi, seed=RNG_SEED):
    '''
    pass in a trajecotry h_seq for 2 dof system 
    set switching and cbf_mode

    then run simulation and get Y, the actual output trajectory
    '''
    # # precompute the 8 possible matrices ONCE (avoid rebuild everytime)
    # modes = {(sf, sb, m): build_mode(sf, sb, m)
    #          for sf in (0, 1) for sb in (0, 1) for m in (MODE_HUMAN, MODE_SAFE)}

    solve = cbf()
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
        sigma_f = int(rng.random() < (p1_f if sigma_f else 1 - p0_f))
        sigma_b = int(rng.random() < (p1_b if sigma_b else 1 - p0_b))

        # 2. AoI, eq (8)-
        aoi_f = aoi(sigma_f, aoi_f)
        aoi_b = aoi(sigma_b, aoi_b)

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
        if cbf_mode == USE_STANDARD_CBF:
            ### here we add in the control barrier function!!
            u_cbf = get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote)
            x_plant = Ap @ x_plant + Bp @ u_cbf #safe
        elif cbf_mode == USE_AOI_CBF:
            u_cbf = get_u_cbf_aoi(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, current_aoi=max(aoi_b, aoi_f))
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




h_seq = generate_human_reference()                              
switching = 1
cbf_mode=1
Ys=[]
Ys_names=[]
for i in [0]:
    for j in [USE_AOI_CBF]:
        switching = i
        cbf_mode = j
        print(cbf_mode, switching)
        rng = np.random.default_rng(RNG_SEED)
        cbf = None if j==USE_NO_CBF else (make_cbf_aoi if j==USE_AOI_CBF else make_unit_circle_cbf)
        Y = simulate_run_cbf(N_s, h_seq, AOI_THRESHOLD, AOI_THRESHOLD, rng, switching, cbf_mode, cbf=cbf)
        Ys.append(Y)
        name = f"actual trajectory [switching={switching}] [cbf_mode={cbf_mode}]"
        Ys_names.append(name)

print("h_seq")
print(h_seq)
for i in range(0,len(Ys)):
    Y=Ys[i]
    plt.plot(Y[0], Y[1], label = f"actual trajectory y - {Ys_names[i]}", alpha=0.5)
plt.plot(h_seq[0], h_seq[1], label = "desired trajectry h")
plt.plot(SAFE_SET_RADIUS*np.cos(np.linspace(0,2*np.pi)-SAFE_SET_CENTER), SAFE_SET_RADIUS*np.sin(np.linspace(0,2*np.pi)-SAFE_SET_CENTER), label='safe set', alpha=0.3)
plt.title("Robot Trajectory vs Desired in R^2")
plt.legend()
plt.show()