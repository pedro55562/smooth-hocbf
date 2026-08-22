# SE(3) velocity control with log-barrier safety terms

This directory is an independent copy of the velocity-control project using a
log-barrier objective instead of explicit CBF-QP inequalities.

## Control Loop

```text
nominal twist -> affine safety functions -> log-barrier NLP -> safe twist xi -> propagate H
```

The decision variable is the spatial twist

```text
xi = [v; omega] in R^6
```

The nominal twist still comes from the SE(3) vector field and the pose is still
propagated with the same spatial-twist convention as the conventional version.

## Optimization Problem

The affine inequalities are built in the same sign convention as the
conventional controller:

```text
A xi >= b
```

The log-barrier solver uses

```text
g_i(xi) = A_i xi - b_i
```

and solves

```text
min_xi 0.5 ||xi - xi_d||^2
       - mu_cbf sum_i log(g_cbf_i(xi))
       - mu_twist_limits sum_j log(g_limit_j(xi))
```

with constant barrier weights:

```text
mu_cbf = 1e-4
mu_twist_limits = 1e-4
```

The CasADi/IPOPT solver is created once before the simulation loop. The previous
solution is reused explicitly as the next initial guess.

## Files

- `main_velocity.py`: simulation loop, warm start and diagnostics
- `constraint_builder.py`: CBF and twist-limit affine functions
- `log_barrier_controller.py`: CasADi/IPOPT log-barrier solver
- `reference_controller.py`: vector-field reference and final-pose regulator
- `se3_utils.py`: SE(3) propagation, path loading and simulation saving
- `scene.py`: robot and obstacle construction
- `plot_results.py`: result plots and diagnostics CSV
- `caminho.txt`: path used by the simulation

Run from this directory:

```bash
python main_velocity.py
```
