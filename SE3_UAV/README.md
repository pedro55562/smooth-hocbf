# SE(3) velocity control with explicit CBF-QP constraints

This version uses direct spatial-twist control.

## Control Loop

```text
nominal twist -> CBF-QP -> safe twist xi -> propagate H directly with xi
```

The CBF remains an explicit QP inequality. No logarithmic barrier is used.

For `h = lambda - lambda_min`, the velocity-level CBF is

```text
D_lambda xi + eta h >= 0
```

or

```text
D_lambda xi >= -eta (lambda - lambda_min)
```

The QP is

```text
min_xi ||xi - xi_d||^2
subject to A_cbf xi >= b_cbf
           -xi_lim <= xi <= xi_lim
```

`VelocityCBFController.solve(..., extra_constraints=...)` already accepts additional conventional linear constraints in the form `(A_i, b_i)` with `A_i @ xi >= b_i`.

## Files

- `main_velocity.py`: simulation loop and configuration
- `cbf_velocity_controller.py`: CBF constraints and QP safety filter
- `reference_controller.py`: vector-field reference and final-pose regulator
- `se3_utils.py`: SE(3) propagation, path loading and error metric
- `scene.py`: robot and obstacle construction
- `plot_results.py`: result plots
- `caminho.txt`: path used by the simulation

Run from this directory:

```bash
python main_velocity.py
```
