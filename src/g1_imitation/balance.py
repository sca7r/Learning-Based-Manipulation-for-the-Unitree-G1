"""Pretrained G1 + Inspire locomotion policy (josabb, mjlab/PPO) used as a standing balance controller."""

import re

import mujoco
import numpy as np
import onnxruntime as ort

from g1_imitation import robot

POLICY_PATH = robot.REPO_ROOT / "assets" / "balance_policy" / "policy.onnx"


class BalancePolicy:
    """Maps robot state and a velocity command to 41 joint position targets.

    Observation layout (159), from the training config:
    base lin vel (3), base ang vel (3), projected gravity (3),
    joint pos - default (53), joint vel (53), last action (41), velocity command (3).
    """

    def __init__(self, model: mujoco.MjModel, cfg: dict):
        self.cfg = cfg["balance"]
        self.session = ort.InferenceSession(str(POLICY_PATH))
        hinges = robot.hinge_joints(model)
        self.qpos_adr = model.jnt_qposadr[hinges]
        self.qvel_adr = model.jnt_dofadr[hinges]
        self.default_pos = robot.default_joint_pos(model)
        self.actuated = model.actuator_trnid[:, 0] - 1  # index of each actuated joint among the 53
        self.scale = robot.action_scale(model)
        self.hidden = np.array([any(re.fullmatch(p, n) for p in self.cfg["hidden_joints"])
                                for n in robot.hinge_joint_names(model)])
        self.last_action = np.zeros(model.nu)
        self.settle_steps = round(self.cfg["settle_time"] / robot.CONTROL_DT)
        self.steps = 0
        self.home = None  # (x, y, yaw) to hold

    @staticmethod
    def base_pose(data: mujoco.MjData) -> tuple[float, float, float]:
        """Pelvis position on the floor and heading (x, y, yaw)."""
        rot = data.xmat[1].reshape(3, 3)
        return data.qpos[0], data.qpos[1], np.arctan2(rot[1, 0], rot[0, 0])

    def hold_command(self, data: mujoco.MjData) -> np.ndarray:
        """Velocity command that keeps the robot where it settled after reset, so it doesn't drift or turn."""
        self.steps += 1
        if self.home is None or self.steps <= self.settle_steps:
            self.home = self.base_pose(data)
            return np.zeros(3)
        x, y, yaw = self.base_pose(data)
        hx, hy, hyaw = self.home
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = c * (hx - x) + s * (hy - y), -s * (hx - x) + c * (hy - y)  # error in the robot's heading frame
        dyaw = (hyaw - yaw + np.pi) % (2 * np.pi) - np.pi
        return np.clip(self.cfg["hold_gain"] * np.array([dx, dy, dyaw]), -self.cfg["hold_limit"], self.cfg["hold_limit"])

    def __call__(self, data: mujoco.MjData, command) -> np.ndarray:
        """Joint position targets for all actuators, for one 50 Hz control step."""
        rot = data.xmat[1].reshape(3, 3)  # pelvis orientation
        # ponytail: velocities at the pelvis origin, not the IMU site a few cm away; noise during training covers it.
        lin_vel = rot.T @ data.qvel[0:3]
        ang_vel = data.qvel[3:6]
        gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        joint_pos = data.qpos[self.qpos_adr] - self.default_pos
        joint_vel = data.qvel[self.qvel_adr].copy()
        # hidden joints (hands, and the right arm we drive ourselves) look still to the policy, so it doesn't react
        joint_pos[self.hidden] = 0.0
        joint_vel[self.hidden] = 0.0

        obs = np.concatenate([lin_vel, ang_vel, gravity, joint_pos, joint_vel, self.last_action, command])
        action = self.session.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
        self.last_action = action
        return self.default_pos[self.actuated] + self.scale * action
