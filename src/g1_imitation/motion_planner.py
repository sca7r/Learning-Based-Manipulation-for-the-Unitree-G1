"""Collision-free right-arm paths through free space (OMPL RRT-Connect), e.g. from the ready pose to the start of
the manipulation.

Plans over the 7 right-arm joints with the rest of the robot frozen where it is. A configuration is valid when the
forearm and hand keep at least [planner] clearance from everything: table, object and the robot's own body.
Collisions are checked by MuJoCo on a planning copy of the model whose forearm and hand geoms carry that clearance
as their contact margin, so any contact it reports means "too close".
"""

import copy

import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from g1_imitation import robot

ou.setLogLevel(ou.LogLevel.LOG_WARN)
ou.RNG.setSeed(1)  # reproducible plans, so success rates can be compared run to run


class ArmPlanner:
    def __init__(self, model: mujoco.MjModel, cfg: dict):
        self.cfg = cfg["planner"]
        self.model = copy.deepcopy(model)
        self.data = mujoco.MjData(self.model)
        joints = [model.joint(n).id for n in robot.RIGHT_ARM_JOINTS]
        self.qpos_adr = model.jnt_qposadr[joints]
        self.lower, self.upper = model.jnt_range[joints].T
        names = [self.model.body(self.model.geom_bodyid[g]).name for g in range(self.model.ngeom)]
        self.moving = np.array([n.startswith(robot.FOREARM_AND_HAND) for n in names])
        self.arm = np.array([n.startswith(robot.FOREARM_AND_HAND + robot.UPPER_ARM) for n in names])
        self.model.geom_margin[self.moving] = self.cfg["clearance"]

    def valid(self, arm_q: np.ndarray) -> bool:
        """True if the forearm and hand keep their clearance from everything except the arm itself."""
        self.data.qpos[self.qpos_adr] = arm_q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        for c in self.data.contact[:self.data.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (self.moving[g1] or self.moving[g2]) and not (self.arm[g1] and self.arm[g2]):
                return False
        return True

    def plan(self, qpos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Joint-space waypoints (N x 7) from the arm configuration in qpos to goal, or raises RuntimeError."""
        self.data.qpos[:] = qpos
        start = qpos[self.qpos_adr]
        for name, q in (("start", start), ("goal", goal)):
            if not self.valid(q):
                raise RuntimeError(f"the arm's {name} configuration is in collision")

        space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for i in range(7):
            bounds.setLow(i, self.lower[i])
            bounds.setHigh(i, self.upper[i])
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)
        setup.setStateValidityChecker(lambda s: self.valid(np.array([s[i] for i in range(7)])))
        setup.getSpaceInformation().setStateValidityCheckingResolution(self.cfg["check_resolution"])
        start_state, goal_state = space.allocState(), space.allocState()
        for i in range(7):
            start_state[i], goal_state[i] = start[i], goal[i]
        setup.setStartAndGoalStates(start_state, goal_state)
        setup.setPlanner(og.RRTConnect(setup.getSpaceInformation()))
        if not setup.solve(self.cfg["timeout"]) or not setup.haveExactSolutionPath():
            raise RuntimeError("no collision-free arm path found")
        setup.simplifySolution()
        path = setup.getSolutionPath()
        return np.array([[path.getState(k)[i] for i in range(7)] for k in range(path.getStateCount())])


def timed(waypoints: np.ndarray, joint_speed: float, dt: float) -> np.ndarray:
    """Joint targets for every control step along the waypoints, no joint moving faster than joint_speed."""
    steps = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / (joint_speed * dt))))
        steps.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(steps)


if __name__ == "__main__":
    # Self-check: plan from the ready pose to the first pick-and-place waypoint, and check every step is valid.
    from g1_imitation import config, scene
    from g1_imitation.arm_ik import RightArmIK

    cfg = config.load()
    model = scene.build_model(cfg)
    data = mujoco.MjData(model)
    scene.reset(model, data, cfg)
    first = cfg["pick_place"]["waypoints"][0]
    target = (scene.object_pos(model, data) - first["back"] * scene.palm_normal(cfg)
              + [0, 0, cfg["grasp"]["height_above_center"] + first["up"]])
    goal, error = RightArmIK(model, cfg).solve(data.qpos, target, scene.grasp_rotation(cfg), iters=200)
    planner = ArmPlanner(model, cfg)
    path = timed(planner.plan(data.qpos.copy(), goal), cfg["planner"]["joint_speed"], robot.CONTROL_DT)
    assert error < 0.01 and all(planner.valid(q) for q in path)
    print(f"ok, {len(path)} steps ({len(path) * robot.CONTROL_DT:.1f} s), every one collision-free")
