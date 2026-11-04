"""Inverse kinematics for the right arm, with mink (QP-based differential IK on MuJoCo).

Solves for the 7 right-arm joints only, everything else frozen at its current state, subject to joint limits
and a minimum clearance between the forearm/hand and both the rest of the robot and the table. The object is left
out: the hand has to touch it.
"""

import mink
import mujoco
import numpy as np

from g1_imitation import robot

class RightArmIK:
    def __init__(self, model: mujoco.MjModel, cfg: dict):
        self.cfg = cfg["ik"]
        joints = [model.joint(n).id for n in robot.RIGHT_ARM_JOINTS]
        self.qpos_adr = model.jnt_qposadr[joints]
        self.actuators = [model.actuator(n).id for n in robot.RIGHT_ARM_JOINTS]
        self.site = model.site("right_grasp").id
        hinge_names = robot.hinge_joint_names(model)
        self.posture = robot.default_joint_pos(model)[[hinge_names.index(n) for n in robot.RIGHT_ARM_JOINTS]]
        limited = robot.hinge_joints(model)[model.jnt_limited[robot.hinge_joints(model)].astype(bool)]
        self.limited_adr = model.jnt_qposadr[limited]
        self.limited_lo, self.limited_hi = model.jnt_range[limited].T

        pelvis = model.body("pelvis").id
        robot_geoms = [g for g in range(model.ngeom)
                       if model.geom_contype[g] and model.body_rootid[model.geom_bodyid[g]] == pelvis]
        names = {g: model.body(model.geom_bodyid[g]).name for g in robot_geoms}
        arm = [g for g in robot_geoms if names[g].startswith(robot.FOREARM_AND_HAND)]
        # The upper arm is always next to the elbow, so it isn't an obstacle for the forearm.
        rest = [g for g in robot_geoms if not names[g].startswith(robot.FOREARM_AND_HAND + robot.UPPER_ARM)]

        self.configuration = mink.Configuration(model)
        self.grasp_task = mink.FrameTask("right_grasp", "site", position_cost=1.0,
                                         orientation_cost=self.cfg["rotation_weight"])
        self.posture_task = mink.PostureTask(model, cost=self.cfg["posture_weight"])
        arm_dofs = set(model.jnt_dofadr[joints])
        self.freeze = mink.DofFreezingTask(model, dof_indices=[i for i in range(model.nv) if i not in arm_dofs])
        self.limits = [
            mink.ConfigurationLimit(model),
            # Bounds each solver step, so obstacles enter the collision check's range before the arm reaches them.
            mink.VelocityLimit(model, {n: self.cfg["joint_speed_limit"] for n in robot.RIGHT_ARM_JOINTS}),
            mink.CollisionAvoidanceLimit(model, geom_pairs=[(arm, rest + [model.geom("table").id])],
                                         minimum_distance_from_collisions=self.cfg["body_clearance"],
                                         collision_detection_distance=self.cfg["collision_check_distance"]),
        ]

    def solve(self, qpos, target_pos, target_rot, iters=None):
        """Returns (7 joint angles, position error [m]). target_rot: 3x3 world rotation of the grasp site."""
        q = np.array(qpos)
        # Contacts can push a frozen joint slightly past its limit, which would make the limit constraint
        # infeasible; the IK sees those joints clamped into range.
        q[self.limited_adr] = np.clip(q[self.limited_adr], self.limited_lo, self.limited_hi)
        self.configuration.update(q)
        q[self.qpos_adr] = self.posture
        self.posture_task.set_target(q)
        self.grasp_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target_rot), target_pos))

        for _ in range(iters or self.cfg["iterations_per_step"]):
            velocity = mink.solve_ik(self.configuration, [self.grasp_task, self.posture_task], robot.CONTROL_DT,
                                     "daqp", damping=self.cfg["damping"], limits=self.limits,
                                     constraints=[self.freeze])
            self.configuration.integrate_inplace(velocity, robot.CONTROL_DT)
        reached = self.configuration.data.site_xpos[self.site]
        return self.configuration.q[self.qpos_adr].copy(), float(np.linalg.norm(target_pos - reached))


if __name__ == "__main__":
    # Self-check: a reachable side-grasp pose in front of the right shoulder is solved to within 1 cm.
    from g1_imitation import config, scene

    cfg = config.load()
    model = scene.build_model(cfg)
    data = mujoco.MjData(model)
    scene.reset(model, data, cfg)
    _, error = RightArmIK(model, cfg).solve(data.qpos, np.array([0.30, -0.2, 0.85]), scene.grasp_rotation(cfg), iters=50)
    assert error < 0.01, error
    print(f"ok, error {1000 * error:.1f} mm")
