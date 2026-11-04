"""Unitree G1 (29 DoF) + Inspire hands in MuJoCo, configured to match the balance policy's training setup.

Gains, armature, default pose and action scales are copied from
assets/balance_policy/env.yaml (the mjlab config the policy was trained with).
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DIR = REPO_ROOT / "assets" / "g1_inspire"
MJCF_PATH = ROBOT_DIR / "g1_inspire.xml"
URDF_PATH = ROBOT_DIR / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"

# joint regex: (stiffness, damping, effort limit [Nm], armature)
ACTUATORS = {
    r".*_(elbow|shoulder_pitch|shoulder_roll|shoulder_yaw|wrist_roll)_joint": (14.2506, 0.9072, 25.0, 0.003609725),
    r".*_(hip_pitch|hip_yaw)_joint|waist_yaw_joint": (40.1792, 2.5579, 88.0, 0.010177520),
    r".*_(hip_roll|knee)_joint": (99.0984, 6.3088, 139.0, 0.025101925),
    r".*_wrist_(pitch|yaw)_joint": (16.7783, 1.0681, 5.0, 0.00425),
    r"waist_(pitch|roll)_joint|.*_ankle_(pitch|roll)_joint": (28.5012, 1.8144, 50.0, 0.00721945),
    r"[LR]_(thumb_proximal_yaw|thumb_proximal_pitch|index_proximal|middle_proximal|ring_proximal|pinky_proximal)_joint":
        (2.0, 0.05, 1.0, 0.0),
}
# The policy's config has zero finger armature, but 4 g finger links then blow up at a 5 ms step.
# 0.01 is what supermimic uses for this hand.
FINGER_ARMATURE = 0.01

# Standing pose the balance policy was trained around (joints not listed: 0)
DEFAULT_JOINT_POS = {
    r".*_hip_pitch_joint": -0.312,
    r".*_knee_joint": 0.669,
    r".*_ankle_pitch_joint": -0.363,
    r".*_elbow_joint": 0.6,
    r"left_shoulder_roll_joint": 0.2,
    r"left_shoulder_pitch_joint": 0.2,
    r"right_shoulder_roll_joint": -0.2,
    r"right_shoulder_pitch_joint": 0.2,
}

# Policy output -> joint target offset, per joint group (from the training config)
ACTION_SCALE = {
    r".*_(elbow|shoulder_pitch|shoulder_roll|shoulder_yaw|wrist_roll)_joint|waist_(pitch|roll)_joint|.*_ankle_(pitch|roll)_joint": 0.43858,
    r".*_(hip_pitch|hip_yaw)_joint|waist_yaw_joint": 0.54755,
    r".*_(hip_roll|knee)_joint": 0.35066,
    r".*_wrist_(pitch|yaw)_joint": 0.07450,
    r"[LR]_.*_proximal.*_joint": 0.125,
}

# Foot contact used in training: 7 capsules per sole, friction 0.6 (mjlab's G1 model).
# The mesh soles in our MJCF made the policy step and turn more.
FOOT_CAPSULES = [  # (x1, y1, x2, y2) in the ankle_roll_link frame, z = -0.025
    (0.100, -0.026, 0.050, -0.026), (-0.044, -0.018, 0.123, -0.018), (-0.052, -0.010, 0.130, -0.010),
    (-0.054, 0.000, 0.132, 0.000), (-0.052, 0.010, 0.130, 0.010), (-0.044, 0.018, 0.123, 0.018),
    (0.100, 0.026, 0.050, 0.026),
]

STANDING_HEIGHT = 0.76  # pelvis height at reset [m]
CONTROL_DT = 0.02  # policy runs at 50 Hz
SIM_DT = 0.005  # 4 physics steps per policy step


RIGHT_ARM_JOINTS = [f"right_{j}_joint" for j in
                    ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
RIGHT_HAND_DRIVES = [f"R_{j}_joint" for j in ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal",
                                               "middle_proximal", "ring_proximal", "pinky_proximal")]
FOREARM_AND_HAND = ("right_elbow", "right_wrist", "R_")  # right arm body name prefixes: the parts that move most
UPPER_ARM = ("right_shoulder",)


def _lookup(table: dict, name: str, default=None):
    """The value of the first regex in `table` that matches the whole joint name."""
    return next((v for pat, v in table.items() if re.fullmatch(pat, name)), default)


def build_spec() -> mujoco.MjSpec:
    """The robot model, set up as the balance policy was trained: physics options, foot contacts, finger couplings
    and position actuators."""
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Elliptic friction cones with friction 10x "harder" than the normal direction (impratio): MuJoCo's documented fix
    # for slow slippage (modeling docs, "Slow slippage"). With pyramidal cones and impratio 1 a held cylinder crept
    # 4-6 cm through the fingers although friction mu*N was 4-10x its weight; with this, 0.6 cm (2 Nov).
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.iterations = 10
    spec.option.ls_iterations = 20

    # Self-collision is on, as in training. The thumb's first link mesh sits inside the palm mesh by
    # construction (they share the thumb mount), so that pair alone is excluded.
    for side, wrist in (("L", "left"), ("R", "right")):
        spec.add_exclude(bodyname1=f"{side}_thumb_proximal", bodyname2=f"{wrist}_wrist_yaw_link")

    for side in ("left", "right"):
        foot = spec.body(f"{side}_ankle_roll_link")
        for geom in foot.geoms:  # the sole meshes stay visible but no longer touch the floor
            geom.contype = 0
        for x1, y1, x2, y2 in FOOT_CAPSULES:
            foot.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.01, 0, 0], fromto=[x1, y1, -0.025, x2, y2, -0.025],
                          friction=[0.6, 0.005, 0.0001], condim=3, priority=1, group=3)

    # Inspire fingers: each intermediate/distal joint follows its proximal driver (URDF <mimic> tags).
    for joint in ET.parse(URDF_PATH).getroot().iter("joint"):
        if (mimic := joint.find("mimic")) is not None:
            eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                                   name1=joint.get("name"), name2=mimic.get("joint"))
            eq.data[:2] = [float(mimic.get("offset", 0)), float(mimic.get("multiplier", 1))]
            spec.joint(joint.get("name")).armature = FINGER_ARMATURE

    # Position actuators in joint definition order, so ctrl[i] lines up with policy action[i].
    for joint in spec.joints:
        gains = _lookup(ACTUATORS, joint.name)
        if gains is None:
            continue
        kp, kd, effort, armature = gains
        joint.armature = FINGER_ARMATURE if joint.name[:2] in ("L_", "R_") else armature
        act = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=kd)
        act.forcelimited = True
        act.forcerange = [-effort, effort]

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            rgba=[0.8, 0.8, 0.8, 1])
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    # The right arm is position-controlled by us with soft gains, so it sags under gravity.
    # Gravity compensation routed through its motors (like a real arm controller) fixes that;
    # the torque still reacts on the body, so balance sees the arm's true weight.
    for body in spec.bodies:
        if body.name.startswith(("right_shoulder", "right_elbow", "right_wrist", "R_")):
            body.gravcomp = 1.0
    for name in RIGHT_ARM_JOINTS:
        spec.joint(name).actgravcomp = True
    return spec


def hinge_joints(model: mujoco.MjModel) -> np.ndarray:
    """Ids of the robot's 53 hinge joints, in definition order (skips free joints of the robot and objects)."""
    return np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)


def hinge_joint_names(model: mujoco.MjModel) -> list[str]:
    """Names of the hinge joints, in the order of hinge_joints."""
    return [model.joint(i).name for i in hinge_joints(model)]


def default_joint_pos(model: mujoco.MjModel) -> np.ndarray:
    """The standing pose (DEFAULT_JOINT_POS) per hinge joint."""
    return np.array([_lookup(DEFAULT_JOINT_POS, n, 0.0) for n in hinge_joint_names(model)])


def action_scale(model: mujoco.MjModel) -> np.ndarray:
    """ACTION_SCALE per actuator."""
    return np.array([_lookup(ACTION_SCALE, model.actuator(i).name) for i in range(model.nu)])


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Robot standing upright at the origin in its default pose, actuators holding that pose."""
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = [0, 0, STANDING_HEIGHT, 1, 0, 0, 0]
    data.qpos[model.jnt_qposadr[hinge_joints(model)]] = default_joint_pos(model)
    # actuator -> its joint id; minus 1 because joint 0 is the robot's free joint, so hinge i is joint i + 1
    data.ctrl[:] = default_joint_pos(model)[model.actuator_trnid[:, 0] - 1]
    mujoco.mj_forward(model, data)
