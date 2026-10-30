from experiment_go import *
from building_block import *
from cbf_layer import *
from cbf_library.mod import instantiate_cbf

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import CheckButtons


def generate_human_reference(N=N_s, radius=HUMAN_DESIRED_TRAJ_RADIUS, distance_from_boundary=SAFE_SET_EPSILON, turns =1):
    '''
    human ref = parameterized curve in R^2, < r(t) cos(t), r(t) sin(t) >
    
    '''
    r = np.linspace(0, radius - distance_from_boundary, N)
    # r = np.array([HUMAN_DESIRED_TRAJ_RADIUS]*N_s)
    t = np.linspace(0, 2 * np.pi * turns, N)
    return np.array([r * np.cos(t), r * np.sin(t)])

def plot_selected_trajectory(Ys, Ys_names, h_seq, SAFE_SET_RADIUS=SAFE_SET_RADIUS, SAFE_SET_CENTER=SAFE_SET_CENTER):
    if not Ys:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.7)

    desired_line, = ax.plot(h_seq[0], h_seq[1], label="desired trajectory h", color="black", alpha=0.5, linewidth=1)

    lines = []
    for Y, name in zip(Ys, Ys_names):
        line, = ax.plot(Y[0], Y[1], alpha=0.15, linewidth=1.2, label=name)
        lines.append((name, line))

    safe_theta = np.linspace(0, 2 * np.pi, 200)
    safe_line, = ax.plot(
        SAFE_SET_RADIUS * np.cos(safe_theta) - SAFE_SET_CENTER,
        SAFE_SET_RADIUS * np.sin(safe_theta) - SAFE_SET_CENTER,
        label='safe set', alpha=0.35, color='tab:green', linewidth=1.5
    )

    checkbox_ax = fig.add_axes([0.73, 0.25, 0.22, 0.5])
    checkboxes = CheckButtons(checkbox_ax, Ys_names)
    selected = {name: True for name in Ys_names}

    def update_plot(_label=None):
        for name, line in lines:
            is_selected = selected.get(name, False)
            line.set_alpha(1.0 if is_selected else 0.12)
            line.set_linewidth(2.5 if is_selected else 1.0)

        active_names = [name for name, is_on in selected.items() if is_on]
        if active_names:
            legend_labels = [*active_names, "desired trajectory h", "safe set"]
            legend_lines = [next(line for n, line in lines if n == name) for name in active_names]
            legend_lines.extend([desired_line, safe_line])
            ax.legend(legend_lines, legend_labels, loc="upper left")
        else:
            ax.legend([desired_line, safe_line], ["desired trajectory h", "safe set"], loc="upper left")
        fig.canvas.draw_idle()

    def on_click(label):
        selected[label] = not selected.get(label, False)
        update_plot(label)

    checkboxes.on_clicked(on_click)
    update_plot()

    plt.show()
    return checkboxes


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
                make_cbf, get_u_cbf = instantiate_cbf(cbf_mode)
                rng = np.random.default_rng(RNG_SEED) #rng must be re-invoked every time for runs to be same rng
                Y = simulate_run_cbf(N_s=N_s, h_seq=h_seq, aoi_threshold_b=AOI_THRESHOLD, aoi_threshold_f=AOI_THRESHOLD, rng=rng, make_cbf=make_cbf, get_u_cbf=get_u_cbf, switching=switching, lossless=lossless)
                   
                Ys.append(Y)
                name = f"lossless={k} switch={switching} cbf={cbf_mode}]"
                Ys_names.append(name)
    print("h_seq")
    print(h_seq)
    plot_selected_trajectory(Ys, Ys_names, h_seq)


if __name__ == "__main__":
    main()
