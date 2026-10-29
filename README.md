# aoi-safety-control

current goal: design an age aware CBF that outperforms the safe mode switching policy.

## quick how-to-run:

1. Edit and save parameters in src/experiment_go.py
2. Run `src/main.py`

## summary of repo organization:

The safe mode switching policy paper is implemented in `standalone_scripts/aoi_based_switching_control_for_safe_haptic_teleoperation_over_a_wireless_network.py`, where the experiments from the paper can be run; this file does not contain any CBF-related implementation and served as me familiarizing myself with the paper.
The equations from the safe mode swithcing policy paper are also listed as functions in the file `src/building_block.py` to be invoked by `src/cbf_layer.py`.

Any CBF-related implementation is developed in `src/cbf_layer.py`; including the convex optimization using the cvxpy library, generation of safe set and human desired reference

And finally, `src/main.py` is the entrypoint to import params from `src/experiment_go.py`, invoke `src/cbf_layer.py` (which invokes `src/building_block.py`), and is responsible for outputting a txt with the outputs from the run, and plotting your trajectories.

So, as a summary:

You can edit the "src/experiment_go.py" file to tweak which experiment you want to run. Then execute the entrypoint 'python src/main.py'. This will run the logic in the src/cbf_layer.py file. Each trajectory takes 2-4 seconds to run.

## requirements

cvxpy, numpy


