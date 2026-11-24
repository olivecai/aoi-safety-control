'''
Instructions:
Edit `src/experiment_go.py`. Save the file. Then `python3 src/main_clf.py`

Runs the local-limit CBF + CLF with a chosen history length (n_x past states, n_u past commands).
'''

from experiment_go import *
from building_block import *
from clf_hist import simulate_run
from cbf_clf_local_hist import make, get_u_cbf

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import CheckButtons

# ---------------------------------------------------------------- new experiment knobs
HIST_SIMULATION = [(1, 1)]      # list of (n_x, n_u) history lengths, e.g. [(0, 0), (1, 1), (3, 3)]
CLF_SIMULATION = [None, "velocity","tracking"]  # None = no CLF; "velocity" = velocity -> 0;
                                                 # "tracking" = position error w.r.t. last received reference
ALPHA = 0.05                    # CLF decay per step (must NOT depend on AoI)
SLACK_WEIGHT = 1.0              # CLF slack penalty (must NOT depend on AoI)
NO_CBF_VALUES = (None, False, 0, "none", "None")   # cbf_mode values meaning "no filter"


def generate_human_reference(N=N_s, radius=HUMAN_DESIRED_TRAJ_RADIUS, distance_from_boundary=SAFE_SET_EPSILON, turns=1):
    '''
    human ref = parameterized curve in R^2, < r(t) cos(t), r(t) sin(t) >
    '''
    r = np.linspace(0, radius - distance_from_boundary, N)
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
            line.set_alpha(0.8 if is_selected else 0.12)
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

    Ys = []
    Ys_names = []
    logs = {}
    for a in AOI_SIMULATION:
        for k in LOSSLESS_SIMULATION:
            for i in SWITCHING_SIMULATION:
                for j in CBF_SIMULATION:
                    for (n_x, n_u) in HIST_SIMULATION:
                        for clf_mode in CLF_SIMULATION:
                            use_clf = clf_mode is not None
                            lossless, switching, cbf_mode, aoi_aware = k, i, j, a
                            no_cbf = cbf_mode in NO_CBF_VALUES

                            # without a filter there is nothing to vary: run it once
                            if no_cbf and (use_clf or (n_x, n_u) != HIST_SIMULATION[0]):
                                continue

                            make_cbf = None if no_cbf else make
                            get_u = None if no_cbf else get_u_cbf
                            rng = np.random.default_rng(RNG_SEED)  # re-invoke every run so runs share the same rng

                            Y = simulate_run(N_s=N_s, h_seq=h_seq, aoi_threshold_b=AOI_THRESHOLD,
                                             aoi_threshold_f=AOI_THRESHOLD, rng=rng,
                                             make_cbf=make_cbf, get_u_cbf=get_u,
                                             switching=switching, lossless=lossless, aoi_aware=aoi_aware,
                                             n_x=n_x, n_u=n_u, use_clf=use_clf,
                                             equilibrium=clf_mode,
                                             alpha=ALPHA, slack_weight=SLACK_WEIGHT)

                            if no_cbf:
                                name = f"lossless={k} switch={switching} no-cbf aoi_aware={aoi_aware}"
                            else:
                                name = (f"lossless={k} switch={switching} cbf={cbf_mode} aoi_aware={aoi_aware} "
                                        f"clf={clf_mode} hist=({n_x},{n_u})")
                            Ys.append(Y)
                            Ys_names.append(name)
                            logs[name] = simulate_run.last_log

                            d = [x for x in simulate_run.last_log["delta"] if x is not None and not np.isnan(x)]
                            if d:
                                print(f"{name}: mean delta={np.mean(d):.4g}, steps with delta>1e-6: "
                                      f"{np.mean(np.array(d) > 1e-6):.1%}, fallbacks: "
                                      f"{sum(bool(f) for f in simulate_run.last_log['fallback'])}")

    print("h_seq")
    print(h_seq)
    plot_selected_trajectory(Ys, Ys_names, h_seq)


if __name__ == "__main__":
    main()