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
y = [C_p x_x, C_p, x_y]T

safe set is R^2 - ||y||^2, R=1
    

CBF_BASE:
u_cbf = mininize over safe u_cbf in D ||u_cbf-u||^2 such that cbf(u_cbf) > 0   

y_(t+1) (u) = [ C_p A_p x_x , C_p A_p x_y ]T + (C_p B_p) u

||y_0 + a u||^2 <= R^2 - (1- gamma)(R^2 - ||y_t||^2) 

'''

from aoi_based_switching_control_for_safe_haptic_teleoperation_over_a_wireless_network import *
import numpy as np
from math import cos, sin

SAFE_SET_RADIUS=1
SAFE_SET_CENTER= 0 
SAFE_SET_EPSILON = 1e-2 
N_s = 1000
seed = 888
AOI_THRESHOLD = 16

def generate_human_reference(N=N_s, radius=SAFE_SET_RADIUS, distance_from_boundary=SAFE_SET_EPSILON):
    '''
    human ref = parameterized curve in R^2, < r(t) cos(t), r(t) sin(t) >
    
    '''
    varying_radius = np.sin(np.linspace(0,radius,N))
    h_x = np.zeros((N))
    h_y = np.zeros((N))
    for i in range(N):
        h_x[i] = varying_radius[i]*cos((2*np.pi*i)/N)
        h_y[i] = varying_radius[i]*sin((2*np.pi*i)/N)
    return [h_x,h_y]

def unit_circle_cbf():
    '''
    u_cbf = argmin over || u_cbf - u ||^2 such that R^2-(y_x_next^2+y_y_next^2) >= (1-gamma)(R^2-(y_x^2+y_y^2))
    y_x_next = Cp @ x_x_next = Cp @ (Ap x_x_next + Bp u_cbf), u_cbf is a function of u, and u = Cc x_x + Dc (h - y)
    '''


def get_u_cbf():
    '''
    
    '''



def simulate_run_cbf(N_s, h_seq, aoi_threshold_f, aoi_threshold_b, rng, switching=True):
    """One random realization. h_seq: reference at each step (length N_s).
    Returns y(t) = Cz z(t)."""
    # # precompute the 8 possible matrices ONCE (avoid rebuild everytime)
    # modes = {(sf, sb, m): build_mode(sf, sb, m)
    #          for sf in (0, 1) for sb in (0, 1) for m in (MODE_HUMAN, MODE_SAFE)}

    z = np.zeros(8) #empty augmented state z(t) = [Xp(t)_T Xc(t)_T Ur(t-1) Yl(t-1)]_T
    sigma_f, sigma_b = 1, 1          # start with both links working
    aoi_f, aoi_b = 1, 1
    y_x = np.zeros(N_s)
    y_y = np.zeros(N_s)

    ##### INITIAL STATE #####
    zeta=0
    Cp = C_plant[zeta]
    Ap = A_plant[zeta]
    Bp = B_plant[zeta]
    Cc = C_controller[zeta]
    Ac = A_controller[zeta]
    Bc = B_controller[zeta]
    Dc = D_controller[zeta]

    x_plant = np.array([[0],[1]]) # stationary 
    y_remote = Cp @ x_plant
    y_local = y_remote # because inthe beginning the aoi is 1
    u_local = Cc @ x_controller + Dc @ ( h - y_local)
    u_remote = u_local # because in the beginning aoi is 1


    for t in range(N_s):
        # 1. channel, eq (4): bursty Markov success/failure
        sigma_f = int(rng.random() < (p1_f if sigma_f else 1 - p0_f))
        sigma_b = int(rng.random() < (p1_b if sigma_b else 1 - p0_b))

        # 2. AoI, eq (8)
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

        h = h_seq[t]

        # system potential update
        x_plant = Ap @ x_plant + Bp @ u_remote
        y_remote = Cp @ x_plant
        y_local = y_remote if sigma_b else y_local #update y_local (local feedback controller side) if the backwards link worked
        x_controller = Ac @ x_controller + Bc @ (h - y_local)
        u_local = Cc @ x_controller + Dc @ ( h - y_local)
        u_remote = u_local if sigma_f else u_remote # update u_remote (remote command on plant side) if the forwards link worked
        
        
        x_x = Ap@x_x + Bp @u
        u = Cp @ x
        u_cbf = get_u_cbf()
        
        

        z = A @ z + B[:, 0] * h_seq[t]



    return y



rng = np.random.default_rng(seed)
h_seq = np.ones(N_s)                               
switching = 0
Y = simulate_run(N_s, h_seq, AOI_THRESHOLD, AOI_THRESHOLD, rng, switching)
print(Y)