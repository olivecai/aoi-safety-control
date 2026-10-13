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

Cz = np.array([15.0, 0.05, 0, 0, 0, 0, 0, 0])

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
    Packet success/failure chain for ONE link (forward or backward).
    State 0 = packet lost, state 1 = packet arrived.
    P[i, j] = P(next state = i | current state = j). Columns sum to 1.
    p0 = P(lost next    | lost now)      "stay failed"
    p1 = P(arrived next | arrived now)   "stay successful"
    '''
    return np.array([[p0,     1 - p1],
                     [1 - p0, p1    ]])

# eq 5
# augmented state z(t) = [xp(t)_T xc(t)_T ur(t-1) yl(t-1)]_T
def update_G(xp, xc, ur_prev, yl_prev, h, sf, sb, m):
    A, B = build_mode(sf=sf, sb=sb, m=m)
    z= np.concatenate([xp, xc, [ur_prev], [yl_prev]]) 
    z_next = A @ z + B[:,0] * h 
    return z_next

#eq 6 eq 7
def build_mode(sf, sb, m): 
    """A_tau, B_tau for tau = (sigma_f, sigma_b, zeta=m). Eq. (6)-(7).
    State z = [x_p (size 2), x_c (size 4), u_r(t-1) (size 1), y_l(t-1) (size 1)]."""
    Ap, Bp, Cp = A_plant[m], B_plant[m], C_plant[m]
    Ac, Bc, Cc, Dc = A_controller[m], B_controller[m], C_controller[m], D_controller[m]
    Z = np.zeros #function can be called by Z(..)
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
      return 1 if (aoi_forward >= aoi_threshold_forwards or aoi_backward >= aoi_threshold_backwards) else 0

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

import scipy.sparse as sp
import scipy.sparse.linalg as spla

# eq 10: list of A_theta for every chain state
def build_mjls(aoi_threshold_f, aoi_threshold_b, switching=True):
    Pi_theta, theta_states = mjls_transition_matrix(p1_f, p0_f, p1_b, p0_b,
                                                    aoi_threshold_f, aoi_threshold_b)
    A_list = []
    for theta in theta_states:
        sf, sb, zeta = f_map(theta, aoi_threshold_f, aoi_threshold_b)
        if not switching:
            zeta = MODE_HUMAN
        A_list.append(build_mode(sf, sb, zeta)[0])
    return Pi_theta, A_list


def drop_unused_state(A_list, idx=3):
    """x_c[1] (z index 3) is an integrator nothing reads. It puts an eigenvalue
    of exactly 1 in every mode, so rho(T) = 1 no matter what. Remove it."""
    keep = [k for k in range(8) if k != idx]
    return [A[np.ix_(keep, keep)] for A in A_list]


# Theorem 1
def spectral_radius_T(Pi_theta, A_list):
    """rho of the second-moment operator. Block (j, i) = Pi[j, i] * kron(A_i, A_i)."""
    N = len(A_list)
    n2 = A_list[0].shape[0] ** 2
    K = [sp.csr_matrix(np.kron(A, A)) for A in A_list]
    blocks = [[None] * N for _ in range(N)]
    for j, i in zip(*np.nonzero(Pi_theta)):
        blocks[j][i] = Pi_theta[j, i] * K[i]
    for j in range(N):                                  # bmat needs >=1 block per row
        if all(b is None for b in blocks[j]):
            blocks[j][j] = sp.csr_matrix((n2, n2))
    T = sp.bmat(blocks, format="csr")
    return abs(spla.eigs(T, k=1, which="LM", return_eigenvectors=False)[0])


def is_mss(aoi_threshold_f, aoi_threshold_b, switching=True):
    Pi_theta, A_list = build_mjls(aoi_threshold_f, aoi_threshold_b, switching)
    rho = spectral_radius_T(Pi_theta, drop_unused_state(A_list))
    return rho < 1, rho
    
# eq 12
def get_prob(m, n, aoi_transition_matrix):
    '''
    P(theta(t+1)=m | theta(t)=n) = Pi_theta[m,n]
    '''
    return aoi_transition_matrix[m,n]

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
    # 
    return c0 * alpha(y_switch) / alpha(y_noswitch) + \
           c1 * (beta(y_switch, eps) / beta(y_noswitch, eps) - 1)

# eq  19 
# human desired haptic reference - stochastic process reflecting intention for haptic reference in repetitive task
# eq 19: transition matrix of the human's desired reference
def human_reference_transition_matrix(N_h, p_stay):
    '''
    Pi_h[i, j] = P(h(t) = h_i | h(t-1) = h_j). Columns sum to 1.
    Repetitive task: stay on the current target, or move to the next one in the cycle.
               now: h0     h1     h2     h3
      next h0     [ p00    p01    p02    p03 ]
      next h1     [ p10    p11    p12    p13 ]
      next h2     [ p20    p21    p22    p23 ]
      next h3     [ p30    p31    p32    p33 ]
    '''
    Pi_h = np.zeros((N_h, N_h))
    for j in range(N_h):
        Pi_h[j, j] = p_stay                  # keep the same target
        Pi_h[(j + 1) % N_h, j] = 1 - p_stay  # move on to the next target
    return Pi_h


# eq 20
# J = Expected[sum from 0 to Ns-1 (w0)norm((Cz(z(t)-z*(t)))) + w1 [vector of 1s ] (z(t) in Xc)]
def evaluation_function(reachable_set, critical_set, z, z_star, length_of_simulation_time, Cz, w0, w1):
    pass



def simulate_run(N_s, h_seq, aoi_threshold_f, aoi_threshold_b, rng, switching=True):
    """One random realization. h_seq: reference at each step (length N_s).
    Returns y(t) = Cz z(t)."""
    # precompute the 8 possible matrices ONCE (avoid rebuild everytime)
    modes = {(sf, sb, m): build_mode(sf, sb, m)
             for sf in (0, 1) for sb in (0, 1) for m in (MODE_HUMAN, MODE_SAFE)}

    z = np.zeros(8) #empty augmented state z(t) = [xp(t)_T xc(t)_T ur(t-1) yl(t-1)]_T
    sigma_f, sigma_b = 1, 1          # start with both links working
    aoi_f, aoi_b = 1, 1
    y = np.zeros(N_s)

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
        A, B = modes[(sf, sb, zeta)]
        y[t] = Cz @ z
        z = A @ z + B[:, 0] * h_seq[t]

    return y

def mc_step_response(aoi_threshold, N_s=1000, runs=500, switching=True, seed=0):
    rng = np.random.default_rng(seed)
    h_seq = np.ones(N_s)                                   # unit step
    Y = np.array([simulate_run(N_s, h_seq, aoi_threshold, aoi_threshold, rng, switching)
                  for _ in range(runs)])
    return Y.mean(axis=0)                                  # ȳ(t), approximately E[Cz z(t)]

def algorithm1(D_range=range(2, 31), runs=500):
    y_noswitch = mc_step_response(2, switching=False, runs=runs)
    results = []
    for D in D_range:
        stable, rho = is_mss(D, D)
        y_switch = mc_step_response(D, runs=runs)
        c = cost(y_noswitch, y_switch)
        results.append((D, c, rho, stable))
        print(f"D={D:2d}  cost={c:.3f}  rho={rho:.5f}  {'stable' if stable else 'UNSTABLE'}")
    feasible = [r for r in results if r[3]]
    best = min(feasible, key=lambda r: r[1])
    print("best:", best)
    return best, results


if __name__ == "__main__":
    print("no switching:", is_mss(16, 16, switching=False))   # expect (False, ~1.0099)
    print("threshold 16:", is_mss(16, 16))                    # expect (True,  ~0.99965)
    algorithm1()