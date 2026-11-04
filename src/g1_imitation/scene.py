"""Tabletop scene: the G1 in front of a table, with an upright cylinder, a goal square and the right hand's grasp point.

All dimensions come from the [scene], [object] and [grasp] sections of config.toml.
"""

import mujoco
import numpy as np

from g1_imitation import robot

PALM = "right_wrist_yaw_link"  # the Inspire palm is part of the last wrist link


def build_model(cfg: dict, object_xy=None, goal_xy=None, object_quat=None) -> mujoco.MjModel:
    return build_spec(cfg, object_xy, goal_xy, object_quat).compile()


def build_spec(cfg: dict, object_xy=None, goal_xy=None, object_quat=None) -> mujoco.MjSpec:
    """The scene before compiling (scripts/showcase.py adds looks to it). object_xy and goal_xy override the configured start and goal positions (randomized trials, per-clip goals);
    object_quat [w, x, y, z]: the cylinder lies on its side in that orientation (stand-up task)."""
    sc, obj, grasp = cfg["scene"], cfg["object"], cfg["grasp"]
    spec = robot.build_spec()
    world = spec.worldbody
    top = sc["table_height"]
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[*sc["table_center"], top / 2], size=[*sc["table_half_size"], top / 2],
                   friction=sc["table_friction"], rgba=[0.55, 0.4, 0.3, 1])
    world.add_geom(name="goal", type=mujoco.mjtGeom.mjGEOM_BOX, contype=0, conaffinity=0,
                   pos=[*(goal_xy if goal_xy is not None else obj["goal"]), top + 0.0005], size=[0.05, 0.05, 0.0005], rgba=[0.2, 0.7, 0.3, 1])

    lying = obj.get("lying", False) or object_quat is not None  # on its side (stand-up task)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (lying_rotation(cfg) @ grasp_rotation(cfg).T if lying else np.eye(3)).ravel())
    if object_quat is not None:
        quat = np.asarray(object_quat, float)
    body = world.add_body(name="object", pos=[*(object_xy if object_xy is not None else obj["start"]),
                                              top + (obj["radius"] if lying else obj["height"] / 2)], quat=quat)
    body.add_freejoint(name="object")
    body.add_geom(name="object", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[obj["radius"], obj["height"] / 2, 0],
                  mass=obj["mass"], friction=obj["friction"], condim=4, rgba=[0.9, 0.2, 0.1, 1])

    for geom in spec.geoms:
        if geom.parent.name == PALM or geom.parent.name.startswith("R_"):
            geom.friction = grasp["hand_friction"]
    for name, limit in grasp["finger_force_limits"].items():
        spec.actuator(name).forcerange = [-limit, limit]

    grasp_point = [grasp["finger_root_x"], grasp["palm_surface_y"] + obj["radius"], grasp["middle_finger_z"]]
    spec.body(PALM).add_site(name="right_grasp", pos=grasp_point, size=[0.01, 0, 0], rgba=[1, 0, 0, 1], group=4)

    # The policy's cameras ([scene] cameras): each on a body, at pos, looking at target (both in that body's frame).
    for name, cam in sc["cameras"].items():
        forward = np.subtract(cam["target"], cam["pos"])
        z = -forward / np.linalg.norm(forward)  # a MuJoCo camera looks along its -z
        x = np.cross(cam["up"], z)
        spec.body(cam["body"]).add_camera(name=name, pos=cam["pos"], fovy=cam["fovy"],
                                          xyaxes=[*x / np.linalg.norm(x), *np.cross(z, x / np.linalg.norm(x))])
    return spec


def reset(model: mujoco.MjModel, data: mujoco.MjData, cfg: dict) -> None:
    """Robot standing at its default pose, except the right arm in its ready pose."""
    robot.reset(model, data)
    for name, angle in cfg["scene"]["right_arm_ready"].items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = angle
        data.ctrl[model.actuator(name).id] = angle
    mujoco.mj_forward(model, data)


def object_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Center of the cylinder."""
    return data.xpos[model.body("object").id].copy()


def placement(cfg: dict, goal_offset, offset) -> tuple[np.ndarray, np.ndarray]:
    """Object start and goal (x, y): the move goal_offset (None: the configured start and goal), centered in
    [episode] workspace and shortened to fit if longer; offset jitters the start."""
    if goal_offset is None:
        return np.array(cfg["object"]["start"]) + offset, np.array(cfg["object"]["goal"])
    lo, hi = (np.array(v) for v in cfg["episode"]["workspace"])
    scale = min(1.0, *(np.divide(hi - lo, np.abs(goal_offset), out=np.full(2, np.inf), where=goal_offset != 0)))
    move = scale * goal_offset
    object_xy = np.clip((lo + hi) / 2 - move / 2 + offset, lo + np.maximum(-move, 0), hi - np.maximum(move, 0))
    return object_xy, object_xy + move


def random_placement(cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Object start and goal (x, y) drawn uniformly over [episode] workspace, at least [episode] random_min_move apart."""
    lo, hi = (np.array(v) for v in cfg["episode"]["workspace"])
    while True:
        start, goal = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(goal - start) >= cfg["episode"]["random_min_move"]:
            return start, goal


def grid_offsets(cfg: dict, n: int) -> np.ndarray:
    """n start offsets for evaluation, evenly spaced over the full [episode] trial_spread forward/back and alternating
    -/+ its sideways extent, so no two trials start at nearly the same place (random draws can repeat)."""
    sx, sy = cfg["episode"]["trial_spread"]
    return np.stack([sx * (np.arange(n) % 2 * 2 - 1), sy * np.linspace(-1, 1, n)], axis=1)


def outcome(model: mujoco.MjModel, data: mujoco.MjData, cfg: dict, goal, max_lift: float, fell: bool) -> dict:
    """How an episode ended; "success": lifted, placed at the goal and upright, let go (the right hand no longer
    touches it), robot standing ([episode])."""
    ep = cfg["episode"]
    final = object_pos(model, data)
    tilt = np.degrees(np.arccos(np.clip(data.xmat[model.body("object").id][8], -1, 1)))  # axis vs vertical
    goal_error = float(np.linalg.norm(final[:2] - np.asarray(goal)[:2]))
    released = not any(b == PALM or b.startswith("R_") for b in touching(model, data))
    success = (max_lift >= ep["lift_success"] and goal_error <= ep["place_tolerance"]
               and tilt <= ep["max_tilt_deg"] and released and not fell)
    return {"lifted": max_lift, "goal_error": goal_error, "tilt_deg": tilt, "released": released, "fell": fell,
            "success": success}


def touching(model: mujoco.MjModel, data: mujoco.MjData) -> set[str]:
    """Names of the bodies in contact with the object: what tactile pads on the hand would feel."""
    obj = model.geom("object").id
    return {model.body(model.geom_bodyid[c.geom2 if c.geom1 == obj else c.geom1]).name
            for c in data.contact[:data.ncon] if obj in (c.geom1, c.geom2)}


def firm_grip(bodies: set[str]) -> bool:
    """Thumb on one side and index or middle finger on the other."""
    return any(b.startswith("R_thumb") for b in bodies) and any(b.startswith(("R_index", "R_middle")) for b in bodies)


def palm_normal(cfg: dict) -> np.ndarray:
    """World direction the palm faces in the side grasp."""
    angle = np.radians(cfg["grasp"]["palm_angle_deg"])
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def lying_rotation(cfg: dict) -> np.ndarray:
    """The side grasp turned 90 degrees about the finger direction so the palm faces down: holding a lying cylinder in
    it, turning back to grasp_rotation stands the cylinder up."""
    turn = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # -90 deg about the grasp frame's x (fingers)
    return grasp_rotation(cfg) @ turn


def grasp_rotation(cfg: dict) -> np.ndarray:
    """World orientation of the grasp point: columns are finger direction, palm normal, index-side axis (up)."""
    normal = palm_normal(cfg)
    up = np.array([0.0, 0.0, 1.0])
    return np.column_stack([np.cross(normal, up), normal, up])
