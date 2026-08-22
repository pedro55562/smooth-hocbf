from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uaibot as ub

from cbf_velocity_controller import CBFConfig, VelocityCBFController
from plot_results import plot_minimum_distance, plot_twist_comparison
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
    cbf_cfg = CBFConfig(
        eta=1.2,
        min_distance=0.01,
        distance_h=0.05,
        distance_eps=0.03,
        twist_limit=1.0,
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

    safety_filter = VelocityCBFController(
        robot_geometry=scene.collision_geometry,
        obstacles=scene.obstacles,
        config=cbf_cfg,
    )

    H = np.asarray(scene.robot.htm, dtype=float)
    path_index = 0
    using_final_regulator = False

    time_history = []
    nominal_twist_history = []
    safe_twist_history = []
    min_distance_history = []
    pose_error_history = []

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
            result = safety_filter.solve(H, nominal_twist)
        except Exception as exc:
            print(f"QP failed at t = {t:.3f} s: {exc}")
            break

        safe_twist = result.twist

        H = propagate_htm(H, safe_twist, sim_cfg.dt)

        scene.robot.add_ani_frame(time=t, htm=H)
        path_marker.add_ani_frame(time=t, htm=htm_path[path_index])

        time_history.append(t)
        nominal_twist_history.append(nominal_twist.copy())
        safe_twist_history.append(safe_twist.copy())
        min_distance_history.append(min(result.distances))
        pose_error_history.append(pose_error_norm(H, H_target))

    save_simulation(sim, base_dir, "se3_velocity")

    plot_minimum_distance(
        min_distance_history,
        time_history,
        safety_distance=cbf_cfg.min_distance,
        output_path=base_dir / "minimum_distance.pdf",
    )

    plot_twist_comparison(
        nominal_twist_history,
        safe_twist_history,
        time_history,
        output_path=base_dir / "twist_comparison.pdf",
    )

    if pose_error_history:
        print(f"Last pose error: {pose_error_history[-1]:.6f}")
        print(f"Minimum recorded distance: {min(min_distance_history):.6f}")


if __name__ == "__main__":
    main()
