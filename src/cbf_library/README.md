This directory contains different CBF functions for you to use in simulation.

We can implement the CBF functions to all satisfy the functions `make()` -> solve: type function and `get_u_cbf(solve, x_plant, u_remote, Ap, Bp, Cp, y_remote, R=SAFE_SET_RADIUS, gamma=0.1, margin=0.0, safe_tolerance=SAFE_TOLERANCE, current_aoi=1):` -> 

If you want to add another CBF, ensure your impl satisfies the two methods `make` and `get_u_cbf` ...(This is kind of like satisfying a trait in Rust, or having a class inherit and implement the methods of its superclass.)