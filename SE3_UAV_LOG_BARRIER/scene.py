from dataclasses import dataclass

import numpy as np
import uaibot as ub


@dataclass
class Scene:
    simulation: object
    robot: object
    collision_geometry: object
    obstacles: list


def create_scene() -> Scene:
    simulation = ub.Simulation.create_sim_hill()

    robot_body = ub.Cylinder(
        htm=ub.Utils.trn([0, 0, 0]) * ub.Utils.roty(np.pi),
        name="robot_body",
        radius=0.3,
        height=0.17,
        color="cyan",
        opacity=0.55,
    )

    robot_model = ub.Model3D(
        url="https://cdn.jsdelivr.net/gh/pedro55562/SE3_CBF_ASSETS@main/TEMA12_DRONA6.obj",
        scale=0.0009,
        mesh_material=ub.MeshMaterial.create_rough_metal(),
    )

    robot_frame = ub.Frame(size=0.10)
    robot_mesh = ub.RigidObject(
        list_model_3d=[robot_model],
        htm=ub.Utils.trn([0, 0, -0.05]) * ub.Utils.roty(np.pi),
    )

    robot = ub.Group(
        list_of_objects=[robot_body, robot_mesh, robot_frame],
        htm=ub.Utils.trn([0, 0, 0.1]) * ub.Utils.roty(np.pi),
    )
    simulation.add([robot])

    material_metal = ub.MeshMaterial.create_rough_metal()
    material_wood = ub.MeshMaterial.create_wood()

    obstacles = [
        ub.Box(
            htm=ub.Utils.trn([0, 2, 0.8]),
            width=3,
            depth=0.1,
            height=1.9,
            mesh_material=material_wood,
        ),
        ub.Box(
            htm=ub.Utils.trn([0, 0, -0.2]),
            width=7,
            depth=7,
            height=0.05,
            mesh_material=material_wood,
        ),
        ub.Box(
            htm=ub.Utils.trn([0, 0, 1.74]),
            width=7,
            depth=7,
            height=0.05,
            mesh_material=material_wood,
        ),
        ub.Box(
            htm=ub.Utils.trn([0, 3.5, 0.8]),
            width=7,
            depth=0.1,
            height=1.9,
            mesh_material=material_wood,
        ),
        ub.Box(
            htm=ub.Utils.trn([-1.5, 2.75, 0.8]) * ub.Utils.rotz(np.pi / 2),
            width=1.5,
            depth=0.1,
            height=1.9,
            mesh_material=material_wood,
        ),
        ub.Box(
            htm=ub.Utils.trn([1.3, 2.42, -0.5]) * ub.Utils.rotz(np.pi / 2),
            width=0.75,
            depth=0.1,
            height=0.95,
            mesh_material=material_metal,
        ),
        ub.Box(
            htm=ub.Utils.trn([1.3, 2.42, 1.52]) * ub.Utils.rotz(np.pi / 2),
            width=0.75,
            depth=0.1,
            height=0.95,
            mesh_material=material_metal,
        ),
        ub.Box(
            htm=ub.Utils.trn([1.3, 3.16, 0.8]) * ub.Utils.rotz(np.pi / 2),
            width=0.74,
            depth=0.1,
            height=1.9,
            mesh_material=material_metal,
        ),
        ub.Cylinder(
            htm=ub.Utils.trn([1.35, 1, 1]),
            height=2,
            radius=0.05,
            mesh_material=material_metal,
        ),
    ]

    simulation.add(obstacles)

    return Scene(
        simulation=simulation,
        robot=robot,
        collision_geometry=robot_body.copy(),
        obstacles=obstacles,
    )
