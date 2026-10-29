
from experiment_go import *
from building_block import *
from cbf_layer import *

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def generate_human_reference(N=N_s, radius=HUMAN_DESIRED_TRAJ_RADIUS, distance_from_boundary=SAFE_SET_EPSILON, turns =1):
    '''
    human ref = parameterized curve in R^2, < r(t) cos(t), r(t) sin(t) >
    
    '''
    r = np.linspace(0, radius - distance_from_boundary, N)
    # r = np.array([HUMAN_DESIRED_TRAJ_RADIUS]*N_s)
    t = np.linspace(0, 2 * np.pi * turns, N)
    return np.array([r * np.cos(t), r * np.sin(t)])

def main():
    h_seq = generate_human_reference()                              

    Ys=[]
    Ys_names=[]
    for k in LOSSLESS_SIMULATION:
        for i in SWITCHING_SIMULATION:
            for j in CBF_SIMULATION:
                lossless=k
                switching = i
                cbf_mode = j
                Y = simulate_run_cbf(N_s, h_seq, AOI_THRESHOLD, AOI_THRESHOLD, rng, switching, cbf_mode, cbf=cbf, lossless=lossless)
                   
                Ys.append(Y)
                name = f"actual trajectory [delay_added={k}] [switching={switching}] [cbf_mode={cbf_mode}]"
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




if __name__ == "__main__":
    main()
