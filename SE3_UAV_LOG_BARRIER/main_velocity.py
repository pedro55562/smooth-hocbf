from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uaibot as ub

from constraint_builder import ConstraintBuilderConfig, VelocityConstraintBuilder
from log_barrier_controller import LogBarrierConfig, LogBarrierController
from plot_results import (
    plot_minimum_distance,
    plot_twist_comparison,
    save_diagnostics_csv,
)
from reference_controller import VectorFieldConfig, pose_regulation_twist, vector_field_twist
from scene import create_scene
from se3_utils import (
    draw_path,
    load_htm_path,
    pose_error_norm,
    propagate_htm,
    save_simulation,
)


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.01
    t_max: float = 40.0
    final_pose_threshold: float = 0.025
    switch_path_fraction: float = 0.72
    final_pose_gain: float = 0.5


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    sim_cfg = SimulationConfig()
    vf_cfg = VectorFieldConfig()
    constraint_cfg = ConstraintBuilderConfig(
        eta=1.2,
        min_distance=0.01,
        distance_h=0.05,
        distance_eps=0.03,
        twist_limit=1.0,
    )
    barrier_cfg = LogBarrierConfig(
        mu_cbf=1e-4,
        mu_twist_limits=1e-4,
        interior_margin=1e-6,
    )

    scene = create_scene()
    sim = scene.simulation

    htm_path = load_htm_path(base_dir / "caminho.txt")
    if not htm_path:
        raise RuntimeError("The path file is empty")

    H_target = np.asarray(htm_path[-1], dtype=float)
    sim.add([ub.Frame(htm=H_target)])
    draw_path(htm_path, sim, color="white", radius=0.02)

    path_marker = ub.Ball(htm=np.eye(4), radius=0.02, color="cyan")
    sim.add([path_marker])

    constraint_builder = VelocityConstraintBuilder(
        robot_geometry=scene.collision_geometry,
        obstacles=scene.obstacles,
        config=constraint_cfg,
    )
    safety_filter = LogBarrierController(
        constraint_count=constraint_builder.constraint_count,
        cbf_count=constraint_builder.cbf_count,
        config=barrier_cfg,
    )

    H = np.asarray(scene.robot.htm, dtype=float)
    path_index = 0
    using_final_regulator = False
    xi_guess = np.zeros((6, 1))

    time_history = []
    nominal_twist_history = []
    safe_twist_history = []
    min_distance_history = []
    pose_error_history = []
    diagnostics = []

    for step in range(int(sim_cfg.t_max / sim_cfg.dt)):
        t = step * sim_cfg.dt
        current_error = pose_error_norm(H, H_target)

        if current_error < sim_cfg.final_pose_threshold:
            print(f"Final pose error: {current_error:.6f}")
            break

        if path_index > sim_cfg.switch_path_fraction * len(htm_path):
            if not using_final_regulator:
                print(f"Switching to final pose regulator at t = {t:.3f} s")
                using_final_regulator = True

            nominal_twist, _ = pose_regulation_twist(
                H,
                H_target,
                gain=sim_cfg.final_pose_gain,
            )
        else:
            nominal_twist, _, path_index = vector_field_twist(H, htm_path, vf_cfg)

        try:
            constraints = constraint_builder.build(H)
            result = safety_filter.solve(nominal_twist, constraints, xi_guess)
        except Exception as exc:
            print(f"Log-barrier solver failed at t = {t:.3f} s: {exc}")
            break

        safe_twist = result.twist
        xi_guess = safe_twist.copy()

        H = propagate_htm(H, safe_twist, sim_cfg.dt)

        scene.robot.add_ani_frame(time=t, htm=H)
        path_marker.add_ani_frame(time=t, htm=htm_path[path_index])

        time_history.append(t)
        nominal_twist_history.append(nominal_twist.copy())
        safe_twist_history.append(safe_twist.copy())
        min_distance_history.append(min(result.distances))
        pose_error_history.append(pose_error_norm(H, H_target))
        diagnostics.append(
            {
                "time": t,
                "solve_time": result.solve_time,
                "solver_iterations": result.solver_iterations,
                "min_slack": result.min_slack,
                "solver_success": result.solver_success,
                "critical_constraint_name": result.critical_constraint_name,
                "nominal_cost": result.nominal_cost,
                "barrier_cost": result.barrier_cost,
                "total_cost": result.total_cost,
            }
        )

    save_simulation(sim, base_dir, "se3_velocity")


    plot_twist_comparison(
        nominal_twist_history,
        safe_twist_history,
        time_history,
        output_path=base_dir / "twist_comparison.pdf",
    )
    save_diagnostics_csv(diagnostics, base_dir / "log_barrier_diagnostics.csv")

    if pose_error_history:
        solve_times = [row["solve_time"] for row in diagnostics]
        iterations = [row["solver_iterations"] for row in diagnostics]
        min_slack_row = min(diagnostics, key=lambda row: row["min_slack"])
        print(f"Solve time: ({np.mean(solve_times):.6f} +/- {np.std(solve_times):.6f})s")
        print(f"Max solve time: {np.max(solve_times):.6f} s")
        print(f"Mean solver iterations: {np.mean(iterations):.2f}")


if __name__ == "__main__":
    main()
