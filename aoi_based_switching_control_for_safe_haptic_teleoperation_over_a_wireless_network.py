'''
Implementation of Algorithm 1 and replication of results from
Y. Yeh, V. S. Varma, S. E. Elayoubi, "AoI-based switching control for safe haptic teleoperation over a wireless network", IEEE CDC 2024.
'''

import numpy as np

MODE_HUMAN = 0
MODE_SAFE = 1

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

# traansmission enviornment: probabilties of successful transmidsion of forward nad backward:
p1_f, p0_f = 0.93, 0.91   # forward:  stay-success, stay-failure
p1_b, p0_b = 0.93, 0.91   # backward: stay-success, stay-failure

# N_s := length of simulation

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
#eq 6
def build_mode(sf, sb, m): #TODO what is this huge A, B, 
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

# eq 11
def f_map(theta, aoi_threshold_forward, aoi_threshold_backward):
    aoi_forward, aoi_backward = theta # simplication is made in paper that aoi backwards and forwards is same

    sigma_f = 1 if aoi_forward == 1 else 0 # sigma_f is 1 if GOOD transmission
    sigma_b = 1 if aoi_backward == 1 else 0# sigma_b is 1 if GOOD transmission

    # but if the aoi is over the threshold for EITHER, then the mode goes into SAFE
    if aoi_forward >= aoi_threshold_forward or aoi_backward >= aoi_threshold_backward:
        zeta = MODE_SAFE
    else:
        zeta = MODE_HUMAN

    return sigma_f, sigma_b, zeta

#eq 12 
# denote transition matrix of the MJLS:
def Pi_aoi_k(p1, p0, aoi_threshold):
      '''
      inputs as p1_f, p0_f, aoi_threshold_forward
      outputs the probability matrix 
      '''
      P = np.zeros((aoi_threshold, aoi_threshold))
      P[0, 0] = p1                    # success -> success
      P[1, 0] = 1 - p1                # success -> first failure
      for j in range(1, aoi_threshold):
            P[0, j] = 1 - p0            # recover -> AoI 1
            P[min(j + 1, aoi_threshold - 1), j] += p0   # keep failing (capped at aoi threshold)
      # example of P where p1 = 0.93, p0 = 0.91, aoi_threshold = 4
      '''
                        now: AoI=1  AoI=2  AoI=3  AoI=4
      next AoI=1  [  0.93   0.09   0.09   0.09 ]   <- packet arrived, reset
      next AoI=2  [  0.07   0      0      0    ]   <- first loss after a success
      next AoI=3  [  0      0.91   0      0    ]   <- lost again
      next AoI=4  [  0      0      0.91   0.91 ]   <- lost again / stuck at cap aoi
      '''
      return P
