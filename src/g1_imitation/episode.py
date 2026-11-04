"""One simulated episode: the G1 stands on its balance policy while its right hand does a manipulation.

1. settle:     the robot stands still, right arm in its ready pose
2. transit:    a collision-free arm path (motion_planner) to the manipulation's first pose
3. manipulate: the hand follows a plan, a function of the plan clock returning
               (grasp point position, grasp point rotation, 6 Inspire drive angles in robot.RIGHT_HAND_DRIVES order)

The plan clock runs with simulated time, except at the optional "grip" event (a plan time, reacting to what the
hand feels): the clock waits there until thumb and index or middle finger hold the object ([episode] grip_timeout).
While the hand holds it, the arm aims higher by its sag ([ik] load_correction_rate).
"""

import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from g1_imitation import motion_planner, robot, scene
from g1_imitation.arm_ik import RightArmIK
from g1_imitation.balance import BalancePolicy


def run(cfg: dict, plan, plan_duration: float, events: dict, object_xy=None, video_path: Path | None = None,
        live: bool = False, goal_xy=None, record: bool = False, object_quat=None, feedback=None) -> dict:
    """feedback(model, data, clock, position, rotation) -> (position, rotation): optional correction of the hand
    target at every control step (scripts/standup_rl.py's learned policy); None leaves the plan as it is.
    record: also return what the policy will see and do, from the first arm move on, every [episode] record_every
    control steps: "qpos" (the whole simulation pose, from which scripts/render_episodes.py draws the head and wrist
    camera images), "state" (right arm joints, then the 6 finger drives) and "action" (their targets)."""
    ep = cfg["episode"]
    model = scene.build_model(cfg, object_xy, goal_xy, object_quat)
    data = mujoco.MjData(model)
    scene.reset(model, data, cfg)
    policy = BalancePolicy(model, cfg)
    ik = RightArmIK(model, cfg)
    fingers = [model.actuator(n).id for n in robot.RIGHT_HAND_DRIVES]
    start = scene.object_pos(model, data)
    goal = np.array([*(goal_xy if goal_xy is not None else cfg["object"]["goal"]), start[2]])
    joints = [model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in [*ik.actuators, *fingers]]
    recorded = {"qpos": [], "state": [], "action": []}

    viewer = mujoco.viewer.launch_passive(model, data) if live else None
    if video_path:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        size = (ep["video"]["width"], ep["video"]["height"])
        renderer = mujoco.Renderer(model, size[1], size[0])
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), ep["video"]["fps"],
                                 (2 * size[0], size[1]))
        camera, closeup = mujoco.MjvCamera(), mujoco.MjvCamera()  # whole scene; hand and object, following the object
        camera.lookat[:] = ep["camera"]["lookat"]
        camera.distance, camera.azimuth, camera.elevation = (
            ep["camera"]["distance"], ep["camera"]["azimuth"], ep["camera"]["elevation"])
        closeup.distance, closeup.azimuth, closeup.elevation = (
            ep["closeup"]["distance"], ep["closeup"]["azimuth"], ep["closeup"]["elevation"])
        steps_per_frame = round(1 / (ep["video"]["fps"] * robot.CONTROL_DT))

    arm_targets = [data.ctrl[ik.actuators].copy()] * round(ep["settle"] / robot.CONTROL_DT)
    finger_targets = np.zeros(len(fingers))
    manipulating, clock, gripped, waited = False, 0.0, False, 0.0
    load_correction, rate = np.zeros(3), cfg["ik"]["load_correction_rate"] * robot.CONTROL_DT
    held = 0.0  # 0-1, follows the firm grip at the same rate; scales [ik] load_feedforward
    # [grasp] stop_at_contact: the listed fingers (support) stop closing where they touch the object, plus
    # contact_squeeze, as a hand curls around an object only until it meets it; the others press to their force limit
    stop = set(cfg["grasp"]["stop_at_contact"])
    finger_q = [model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in fingers]
    finger_part = ["R_" + model.actuator(a).name.split("_")[1] for a in fingers]  # R_index, R_thumb, ...
    palm_touched = False  # [grasp] palm_free: the object must never touch the palm (stand-up)
    step, max_lift, fell, error = 0, 0.0, False, None
    while clock < plan_duration + ep["hold_after"]:
        tick = time.perf_counter()
        data.ctrl[:] = policy(data, policy.hold_command(data))
        if step < len(arm_targets):
            data.ctrl[ik.actuators] = arm_targets[step]
            data.ctrl[fingers] = finger_targets
        elif not manipulating:  # settled: plan the transit to the manipulation's first pose
            position, rotation, finger_targets = plan(0.0)
            entry, _ = ik.solve(data.qpos, position, rotation, iters=200)
            try:
                path = motion_planner.ArmPlanner(model, cfg).plan(data.qpos.copy(), entry)
            except RuntimeError as e:
                error = f"transit planning: {e}"
                break
            arm_targets += list(motion_planner.timed(path, cfg["planner"]["joint_speed"], robot.CONTROL_DT))
            manipulating = True
            data.ctrl[ik.actuators] = arm_targets[step]
        else:
            felt = scene.touching(model, data)
            palm_touched |= scene.PALM in felt
            if not gripped and clock >= events.get("grip", np.inf):
                clock, waited = events["grip"], waited + robot.CONTROL_DT
                gripped = scene.firm_grip(felt)
                if waited > ep["grip_timeout"]:
                    error = f"no firm grip; touching the cylinder: {', '.join(sorted(felt)) or 'nothing'}"
                    break
            position, rotation, finger_targets = plan(clock)
            if feedback is not None:
                position, rotation = feedback(model, data, clock, position, rotation)
            holding = gripped and scene.firm_grip(felt)
            if holding:
                load_correction += rate * (position - data.site("right_grasp").xpos) * cfg["ik"]["load_correction_axes"]
            else:
                load_correction *= 1 - rate
            held += rate * (holding - held)
            data.ctrl[ik.actuators], _ = ik.solve(data.qpos, position + load_correction, rotation)
            if cfg["ik"]["load_feedforward"]:
                data.ctrl[ik.actuators] += held * cfg["ik"]["load_feedforward"] * load_feedforward(model, data, ik)
            # while a listed finger touches the object it closes no further than contact_squeeze past where it is; if
            # the hand moves and it loses contact, it curls on until it meets the object again
            for j, part in enumerate(finger_part):
                if part in stop and any(b.startswith(part) for b in felt):
                    finger_targets[j] = min(finger_targets[j], data.qpos[finger_q[j]] + cfg["grasp"]["contact_squeeze"])
            data.ctrl[fingers] = finger_targets
            clock += robot.CONTROL_DT
        if record and manipulating and step % ep["record_every"] == 0:
            recorded["qpos"].append(data.qpos.copy())
            recorded["state"].append(data.qpos[joints].copy())
            recorded["action"].append(np.r_[data.ctrl[ik.actuators], data.ctrl[fingers]])
        mujoco.mj_step(model, data, nstep=round(robot.CONTROL_DT / robot.SIM_DT))

        max_lift = max(max_lift, scene.object_pos(model, data)[2] - start[2])
        fell |= data.qpos[2] < robot.STANDING_HEIGHT / 2
        if video_path and step % steps_per_frame == 0:
            closeup.lookat[:] = scene.object_pos(model, data)
            frames = []
            for cam in (camera, closeup):
                renderer.update_scene(data, cam)
                frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            frame = np.hstack(frames)
            phase = "transit" if not manipulating else f"plan t={clock:.1f}s" + (" GRIPPED" if gripped else "")
            touch = ", ".join(sorted(b.removeprefix("R_") for b in scene.touching(model, data)))
            cv2.putText(frame, f"{phase}  touching: {touch}", (8, size[1] - 10), 0, 0.5, (0, 0, 0), 1)
            writer.write(frame)
        if viewer:
            viewer.sync()
            time.sleep(max(0.0, robot.CONTROL_DT - (time.perf_counter() - tick)))
        step += 1
    if video_path:
        writer.release()
        renderer.close()  # unclosed, it makes the next trial's images garbage under WSL's GPU driver (d3d12)
    if viewer:
        viewer.close()

    result = scene.outcome(model, data, cfg, goal, max_lift, fell)
    if cfg["grasp"].get("palm_free") and palm_touched and error is None:
        error = "the cylinder touched the palm"
    result["success"] &= error is None
    return {**result, "error": error, "joint_names": [model.actuator(a).name for a in [*ik.actuators, *fingers]],
            **({"frames": {k: np.array(v) for k, v in recorded.items()}} if record else {})}


def load_feedforward(model: mujoco.MjModel, data: mujoco.MjData, ik: RightArmIK) -> np.ndarray:
    """Arm motor target offsets that hold the cylinder's weight: a position motor settles where kp (target - q) balances
    the load torque J^T F, so aiming tau / kp further cancels the sag before it happens."""
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac, None, ik.site)
    dofs = model.jnt_dofadr[model.actuator_trnid[ik.actuators, 0]]
    weight = np.array([0.0, 0.0, model.opt.gravity[2] * model.body_mass[model.body("object").id]])
    return -(jac[:, dofs].T @ weight) / model.actuator_gainprm[ik.actuators, 0]


def waypoint_plan(cfg: dict, object_start: np.ndarray, goal: np.ndarray, waypoints: list | None = None):
    """Plan, duration and events for episode.run: the grasp point moves in straight lines between the waypoints (see
    [pick_place] in config.toml), and the hand changes gradually to each waypoint's pose ([grasp] fingers_<hand>).
    A waypoint's "until" makes the end of its segment an event (see run). waypoints: a subset, default all."""
    waypoints = waypoints or cfg["pick_place"]["waypoints"]
    normal = scene.palm_normal(cfg)
    lift = cfg["grasp"]["height_above_center"]
    rotation = scene.grasp_rotation(cfg)
    hands = [np.array([cfg["grasp"]["fingers_" + wp["hand"]][n] for n in robot.RIGHT_HAND_DRIVES]) for wp in waypoints]
    targets = [(object_start if wp["at"] == "object" else goal) - wp["back"] * normal + [0, 0, lift + wp["up"]]
               - [wp.get("toward", 0.0), 0, 0] for wp in waypoints]  # toward: closer to the robot's body (-x)
    ends = np.cumsum([wp["duration"] for wp in waypoints])
    # grasp rule: while closing, middle, ring and pinky start [grasp] support_delay apart after thumb and index
    lag = np.r_[0, 0, 0, 1, 2, 3] * cfg["grasp"]["support_delay"]

    def plan(t: float):
        i = min(int(np.searchsorted(ends, t, side="right")), len(waypoints) - 1)
        since = t if i == 0 else t - ends[i - 1]
        alpha = 1.0 if i == 0 else min(since / waypoints[i]["duration"], 1.0)
        position = targets[max(i - 1, 0)] + alpha * (targets[i] - targets[max(i - 1, 0)])
        before, after = hands[max(i - 1, 0)], hands[i]
        closing = after > before
        a = np.where(closing, np.clip((since - lag) / waypoints[i]["duration"], 0, 1), alpha) if i else alpha
        return position, rotation, before + a * (after - before)

    return plan, ends[-1], {wp["until"]: end for wp, end in zip(waypoints, ends) if "until" in wp}


def trials(cfg: dict, make_plan, n: int, video_path: Path | None = None, live: bool = False, goal_offset=None,
           record_dir: Path | None = None, seed: int = 0, offsets=None, random_goals: bool = False,
           object_quat=None, task: str = "pick_place") -> int:
    """Runs make_plan(object_start, goal) -> (plan, duration, events) as n episodes and prints each result and the
    success rate; returns the number of successes. With n > 1 the object start is randomized by [episode]
    trial_spread (or given per trial as offsets, e.g. scene.grid_offsets). goal_offset: the goal relative to the start (a demonstration's own move), else the configured
    goal; it is centered in [episode] workspace (shortened if longer). The first [episode] video_trials trials are recorded to video_path, with the number added when there are several
    (pick_place_3.mp4); live: watch each trial in the MuJoCo viewer instead of saving its video. record_dir: each successful trial is saved there
    as training data, <video stem>_<trial>/data.npz (qpos, state, action, fps, start, goal); scripts/render_episodes.py
    then adds head.mp4 and wrist.mp4. random_goals: start and goal both random in the workspace
    (scene.random_placement) instead of goal_offset."""
    rng = np.random.default_rng(seed)
    z = cfg["scene"]["table_height"] + cfg["object"]["height"] / 2
    successes = 0
    for trial in range(n):
        offset = (offsets[trial] if offsets is not None else
                  np.zeros(2) if n == 1 else rng.uniform(-1, 1, 2) * cfg["episode"]["trial_spread"])
        object_xy, goal_xy = (scene.random_placement(cfg, rng) if random_goals else
                              scene.placement(cfg, goal_offset, offset))
        if goal_offset is not None and not random_goals and (scale := np.linalg.norm(goal_xy - object_xy) / np.linalg.norm(goal_offset)) < 1:
            print(f"trial {trial:3d}  the demo's {100 * np.linalg.norm(goal_offset):.0f} cm move is longer than the "
                  f"workspace, scaled by {scale:.2f}")
        plan, duration, events = make_plan(np.array([*object_xy, z]), np.array([*goal_xy, z]))
        single = n == 1
        path = None if live or video_path is None or trial >= cfg["episode"]["video_trials"] else (
            video_path if single else video_path.with_stem(f"{video_path.stem}_{trial}"))
        r = run(cfg, plan, duration, events, object_xy, video_path=path, live=live, goal_xy=goal_xy,
                record=record_dir is not None, object_quat=object_quat)
        successes += r["success"]
        print(report(trial, r))
        if record_dir is not None and r["success"]:
            out = record_dir / f"{(video_path or Path('run')).stem}_{trial}"
            out.mkdir(parents=True, exist_ok=True)
            fps = 1 / (cfg["episode"]["record_every"] * robot.CONTROL_DT)
            np.savez(out / "data.npz", qpos=r["frames"]["qpos"], state=r["frames"]["state"],
                     action=r["frames"]["action"], fps=fps, start=object_xy, goal=goal_xy, names=r["joint_names"],
                     task=task)
    print(f"success rate {successes}/{n}")
    return successes


def report(trial: int, r: dict) -> str:
    if r["error"]:
        return f"trial {trial:3d}  FAIL ({r['error']})"
    return (f"trial {trial:3d}  lifted {100 * r['lifted']:4.1f} cm  goal error {100 * r['goal_error']:4.1f} cm  "
            f"tilt {r['tilt_deg']:4.1f} deg  {'FELL ' if r['fell'] else ''}{'SUCCESS' if r['success'] else 'FAIL'}")
