'''
Implementation of Algorithm 1 and replication of results from
Y. Yeh, V. S. Varma, S. E. Elayoubi, "AoI-based switching control for safe haptic teleoperation over a wireless network", IEEE CDC 2024.
'''

import numpy as np

MODE_HUMAN = 0
MODE_SAFE = 1

# traansmission enviornment: probabilties of successful transmidsion of forward nad backward:
p1_f, p0_f = 0.93, 0.91   # forward:  stay-success, stay-failure
p1_b, p0_b = 0.93, 0.91   # backward: stay-success, stay-failure

# N_s := length of simulation

#eq 1 
'''
SYSTEM MODEL
xp(t+1) = Ap[mode] xp(t) + Bp[mode] ur(t)
yr(t) = Cp[mode] xp(t)
'''
def plant_dynamics(xp, ur, m):
      '''
      xp : plant state vector
      ur : control input vector
      m : mode (HUMAN_MODE ==0 for human, SAFE_MODE ==1for safe)
      returns xp (next plant state vec) and yr (haaptic feedbakc recv from plant)
      '''
      Ap, Bp, Cp = A_plant[m], B_plant[m], C_plant[m]
      xp_next = Ap @ xp + Bp @ ur
      yr = Cp @ xp
      return xp_next, yr

# eq 2
'''
LOCAL CONTROLLER DISCRETE TIME LINEAR SYSTEM
xc(t+1) = Ac[mode] xc(t)  + Bc[mode](h(t)-yl(t))
ul(t) = Cc[mode] xc(t) + Dc[mode] (h(t) - yl(t))
'''
def controller_dynamics(xc, h, yl, m):
    '''
    xc : controller state vector
    h: humaan desired haptic reference
    yl : feedback from plant (yl=yr if sigma_b (transmissino backward link health) is always 1 (good) otherwise HLS)
    m: mode
    returns 
    xc_next : next controller state vector
    ul: the next command outgoing to the plant
    
    '''
    Ac, Bc, Cc, Dc = A_controller[m], B_controller[m], C_controller[m], D_controller[m]
    xc_next = Ac @ xc + Bc @ (h - yl)
    ul = Cc @ xc + Dc @ ( h - yl)
    return xc_next, ul

# eq 3
def HLS(status_of_the_link, ideal_msg_update, hls_update):
    return ideal_msg_update if status_of_the_link else hls_update

# eq 4
def P_sigma(p0, p1):
    '''
    p0, p1 are the probabilities of staying in the same state for a single link (forward or backward)
    p0 means stay in HUMAN MODE
      p1 means stay in SAFE MODE
    '''
    Pi_sigma_k = np.array([[p0, 1-p1],[1-p0, p1]])
    return Pi_sigma_k

# eq 5
# augmented state z(t) = [xp(t)_T xc(t)_T ur(t-1) yl(t-1)]_T
def update_G(xp, xc, ur_prev, yl_prev, h, sf, sb, m):
    A, B = build_mode(sf=sf, sb=sb, m=m)
    z= np.array([xp.reshape(1,-1), xc.reshape(1,-1), ur_prev, yl_prev]) 
    z_next = A @ z + B @ h 
    return z_next

#eq 6 eq 7
def build_mode(sf, sb, m): 
    """A_tau, B_tau for tau = (sigma_f, sigma_b, zeta=m). Eq. (6)-(7).
    State z = [x_p (2), x_c (4), u_r(t-1) (1), y_l(t-1) (1)]."""
    Ap, Bp, Cp = A_plant[m], B_plant[m], C_plant[m]
    Ac, Bc, Cc, Dc = A_controller[m], B_controller[m], C_controller[m], D_controller[m]
    Z = np.zeros
    I1 = np.eye(1)

    if sf and sb:            # (1,1,0): both packets arrive
        A = np.block([
            [Ap - Bp @ Dc @ Cp,  Bp @ Cc,  Z((2, 1)),  Z((2, 1))],
            [-Bc @ Cp,           Ac,       Z((4, 1)),  Z((4, 1))],
            [-Dc @ Cp,           Cc,       Z((1, 1)),  Z((1, 1))],
            [Cp,                 Z((1, 4)), Z((1, 1)), Z((1, 1))]])
        B = np.vstack([Bp @ Dc, Bc, Dc, Z((1, 1))])

    elif sf and not sb:      # (1,0,m): command arrives, feedback lost
        A = np.block([
            [Ap,         Bp @ Cc,   Z((2, 1)),  -Bp @ Dc],
            [Z((4, 2)),  Ac,        Z((4, 1)),  -Bc],
            [Z((1, 2)),  Cc,        Z((1, 1)),  -Dc],
            [Z((1, 2)),  Z((1, 4)), Z((1, 1)),  I1]])
        B = np.vstack([Bp @ Dc, Bc, Dc, Z((1, 1))])

    elif not sf and sb:      # (0,1,m): command lost, feedback arrives
        A = np.block([
            [Ap,         Z((2, 4)), Bp,         Z((2, 1))],
            [-Bc @ Cp,   Ac,        Z((4, 1)),  Z((4, 1))],
            [Z((1, 2)),  Z((1, 4)), I1,         Z((1, 1))],
            [Cp,         Z((1, 4)), Z((1, 1)),  Z((1, 1))]])
        B = np.vstack([Z((2, 1)), Bc, Z((1, 1)), Z((1, 1))])

    else:                    # (0,0,m): both lost
        A = np.block([
            [Ap,         Z((2, 4)), Bp,         Z((2, 1))],
            [Z((4, 2)),  Ac,        Z((4, 1)),  -Bc],
            [Z((1, 2)),  Z((1, 4)), I1,         Z((1, 1))],
            [Z((1, 2)),  Z((1, 4)), Z((1, 1)),  I1]])
        B = np.vstack([Z((2, 1)), Bc, Z((1, 1)), Z((1, 1))])

    return A, B

# eq 8
def aoi(link_status, aoi):
    return 1 if link_status else aoi+1 

# eq 9
def discrete_switching_policy(aoi_forward, aoi_backward, aoi_threshold_forwards, aoi_threshold_backwards):
      return 1 if (aoi_forward > aoi_threshold_forwards or aoi_backward > aoi_threshold_backwards) else 0

# theta for eq 10
def get_theta(aoi_forward, aoi_backward, aoi_threshold_forwards, aoi_threshold_backwards):
    return (min(aoi_forward, aoi_threshold_forwards), min(aoi_backward, aoi_threshold_backwards))

# eq 10
'''
get_theta, then pass theta into f_map
then call eq 5 update_G as usual with the same arguments, but the only difference is sf sb m are from the f map
'''
# eq 11

def f_map(theta, aoi_threshold_forward, aoi_threshold_backward):
    aoi_forward, aoi_backward = theta
    sigma_f = 1 if aoi_forward == 1 else 0 # sigma_f is 1 if GOOD transmission
    sigma_b = 1 if aoi_backward == 1 else 0# sigma_b is 1 if GOOD transmission

    # but if the aoi is over the threshold for EITHER, then the mode goes into SAFE
    if aoi_forward >= aoi_threshold_forward or aoi_backward >= aoi_threshold_backward:
        zeta = MODE_SAFE
    else:
        zeta = MODE_HUMAN

    return sigma_f, sigma_b, zeta

def get_pro

# eq 13: one link's AoI transition matrix
def aoi_transition_matrix(p_stay_success, p_stay_fail, aoi_threshold):
    '''
    p_stay_fail = p0 If the last packet was lost, the chance the next one is also lost
    p_stay_success = p1 If the last packet was received, the chance the next one is also received
    1 - p_stay_fail = p0' If the last packet was lost, the chance the next one is received
    1 - p_stay_success = p1' If the last packet was received, the chance the next one is lost
    aoi_threshold = D The maximum AoI before switching
    '''
    """Pi[i, j] = P(next AoI = i+1 | current AoI = j+1). Index 0 = AoI 1."""
    D = aoi_threshold
    Pi = np.zeros((D, D))
    Pi[0, 0] = p_stay_success          # arrived, arrives again -> AoI 1
    Pi[1, 0] = 1 - p_stay_success      # arrived, then lost     -> AoI 2
    for j in range(1, D):
        Pi[0, j] = 1 - p_stay_fail             # recovers -> AoI 1
        Pi[min(j + 1, D - 1), j] = p_stay_fail # lost again -> AoI+1 (capped)
    '''
      example of Pi, where p_stay_success = 0.93, p_stay_fail = 0.91, aoi_threshold = 4
      [[0.93 0.09 0.09 0.09]
      [0.07 0.   0.   0.  ]
      [0.   0.91 0.   0.  ]
      [0.   0.   0.91 0.91]]

      for a general p_stay_success, p_stay_fail, aoi_threshold, the matrix is:
      [[p_stay_success, 1 - p_stay_fail, 1 - p_stay_fail, ..., 1 - p_stay_fail]
      [1 - p_stay_success, 0, 0, ..., 0]
      [0, p_stay_fail, 0, ..., 0]
      [0, 0, p_stay_fail, ..., 0]
      ...
      [0, 0, 0, ..., p_stay_fail, p_stay_fail]]
      '''
    return Pi


# eq 12: joint chain theta = (forward AoI, backward AoI)
def mjls_transition_matrix(p1_f, p0_f, p1_b, p0_b,
                           aoi_threshold_forward, aoi_threshold_backward):
    """Pi_theta[m, n] = P(theta(t+1) = m | theta(t) = n).
    Returns the matrix and the list of theta states in matching order."""
    Pi_f = aoi_transition_matrix(p1_f, p0_f, aoi_threshold_forward)
    Pi_b = aoi_transition_matrix(p1_b, p0_b, aoi_threshold_backward)
    Pi_theta = np.kron(Pi_f, Pi_b) #kronecker product of the two AoI transition matrices to get the joint transition matrix

    # state order must match np.kron: forward AoI outer, backward AoI inner
    theta_states = [(af, ab)
                    for af in range(1, aoi_threshold_forward + 1)
                    for ab in range(1, aoi_threshold_backward + 1)]
    return Pi_theta, theta_states

# eq 16
A_plant = {MODE_HUMAN: np.array([[0.9982, 0.0009463], [-3.666, 0.8926]]),
      MODE_SAFE: np.array([[0.9999, 0.0009658], [-0.1448, 0.9317]])}
B_plant = {MODE_HUMAN: np.array([[0.001758], [3.516]]),
           MODE_SAFE:  np.zeros((2, 1))}                   
C_plant = {MODE_HUMAN: np.array([[15.0, 0.05]]), 
      MODE_SAFE: np.array([[15.0, 0.05]])}
D_plant = {MODE_HUMAN: np.zeros((1, 1)), MODE_SAFE: np.zeros((1, 1))}

# eq 17 
A_controller = {MODE_HUMAN: np.array([[0, 0, 0, 0], [0, 1, 0, 0],
                   [-0.005, 0, 1, 0.0009], [-0.9063, 0, 0, 0.8187]]),
      MODE_SAFE: np.array([[0, 0, 0, 0], [0, 1, 0, 0],
                   [-0.005, 0, 1, 0.0006], [-0.6321, 0, 0, 0.3679]])}
B_controller = {MODE_HUMAN: np.array([[1], [0.01], [0.008], [1.5408]]),
      MODE_SAFE: np.array([[1], [0.01], [0.006], [1.0746]])}
C_controller = {MODE_HUMAN: np.array([[0, 0, 1, 0]]), 
      MODE_SAFE: np.array([[0, 0, 1, 0]])}
D_controller = {MODE_HUMAN: np.zeros((1, 1)), MODE_SAFE: np.zeros((1, 1))}

# to decide optimal AOI threshold: need 2 metrics:
# 1. suppression of magnitude overshoot
# 2. speed of response
def alpha(y):                     # peak magnitude
    return np.max(np.abs(y))
 
def beta(y, eps=0.05):            # first time within 1 +- eps
    idx = np.nonzero(np.abs(y - 1) <= eps)[0]
    return idx[0] if idx.size else len(y)

# eq 18
def cost(y_noswitch, y_switch, c0=7, c1=1, eps=0.05):
    return c0 * alpha(y_switch) / alpha(y_noswitch) + \
           c1 * (beta(y_switch, eps) / beta(y_noswitch, eps) - 1)
