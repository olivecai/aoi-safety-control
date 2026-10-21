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
AOI_THRESHOLD = 13

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
    u_cbf, status = solve(u_remote.ravel(), CpApxp, CpBp, rho)
    return np.asarray(u_cbf).reshape(1,2)
    

def simulate_run_cbf(N_s, h_seq, aoi_threshold_f, aoi_threshold_b, rng, switching=True, cbf_mode=True, cbf=make_unit_circle_cbf, seed=RNG_SEED):
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
    x_plant = np.zeros((2, 2))  # stationary 
    
    h = h_seq[:,0]
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
        if cbf_mode:
            ### here we add in the control barrier function!!
            u_cbf = get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote)
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
for i in [0,1]:
    for j in [0,1]:
        switching = i
        cbf_mode = j
        rng = np.random.default_rng(RNG_SEED)
        Y = simulate_run_cbf(N_s, h_seq, AOI_THRESHOLD, AOI_THRESHOLD, rng, switching, cbf_mode)
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