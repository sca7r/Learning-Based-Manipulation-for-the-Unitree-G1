"""Runs a trained LeRobot policy (ACT or Diffusion Policy) on the simulated G1 and reports its success rate, with the
same start/goal placement and success test as data generation (scene.placement, scene.outcome).

  python scripts/evaluate.py outputs/act_v3/checkpoints/400000/pretrained_model --trials 5
  python scripts/evaluate.py CHECKPOINT --start 0.30 -0.25 --goal 0.35 -0.12 --live   # your own positions, in the viewer
  python scripts/evaluate.py CHECKPOINT --random-goals --trials 10                       # start and goal anywhere in reach

Positions are (x forward, y left) in metres from the robot's pelvis; the arm reaches [episode] workspace
(x 0.25..0.40, y -0.32..-0.10).

Each trial: the robot settles in its ready pose ([episode] settle), then the policy alone drives the right arm and the
6 finger drives from the head and wrist cameras and the joint state, at the training data's rate ([episode]
record_every), for [evaluate] seconds; the balance policy keeps the robot standing. Goals: each clip's move, read from
its training episodes (data.npz start/goal); starts are new random draws (--seed; generation used 0), drawn per clip
exactly as replay_demos.py does, so `replay_demos.py CLIP --trials N --seed S` runs the expert on the same starts.
Needs lerobot and mujoco but not OMPL, so it runs natively on Windows (fast OpenGL), next to the GPU.
Writes outputs/evaluate/<clip>_<trial>.mp4 (scene | head camera | wrist camera) for the first --videos trials per clip,
and <clip>_<trial>.npz for every trial: the policy's joint targets (action), joint positions (state) and the cylinder's
position (object) at 25 Hz, in
the same layout as the training data, and the outcome, for comparing smoothness and success across runs.
"""

import argparse
import ast
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from g1_imitation import config, robot, scene
from g1_imitation.balance import BalancePolicy
from to_lerobot import TASK


def rollout(cfg, policy, device, pre, post, object_xy, goal_xy, video_path=None, live=False, object_quat=None) -> dict:
    """One trial with the policy in control of the right arm and hand: the outcome (scene.outcome) plus the logged
    states, actions, cylinder positions and full poses."""
    ep = cfg["episode"]
    model = scene.build_model(cfg, object_xy, goal_xy, object_quat)
    data = mujoco.MjData(model)
    scene.reset(model, data, cfg)
    balance = BalancePolicy(model, cfg)
    arm = [model.actuator(n).id for n in robot.RIGHT_ARM_JOINTS]
    fingers = [model.actuator(n).id for n in robot.RIGHT_HAND_DRIVES]
    joints = [model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in [*arm, *fingers]]  # as in episode.run's state
    width, height = ep["record_size"]
    eye = mujoco.Renderer(model, height, width)
    view = mujoco.MjvCamera()
    view.lookat[:] = ep["camera"]["lookat"]
    view.distance, view.azimuth, view.elevation = ep["camera"]["distance"], ep["camera"]["azimuth"], ep["camera"]["elevation"]
    writer = video_path and cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                            1 / (ep["record_every"] * robot.CONTROL_DT),
                                            (3 * width, height))
    start = scene.object_pos(model, data)
    action = np.r_[data.ctrl[arm], np.zeros(len(fingers))]  # ready pose, open hand: what the settle phase holds
    settle = round(ep["settle"] / robot.CONTROL_DT)
    policy.reset()
    max_lift, fell = 0.0, False
    log = {"state": [], "action": [], "object": [], "qpos": []}  # qpos: re-renders the images (to_lerobot.frames)
    viewer = live and mujoco.viewer.launch_passive(model, data)
    for step in range(settle + round(cfg["evaluate"]["seconds"] / robot.CONTROL_DT)):
        data.ctrl[:] = balance(data, balance.hold_command(data))  # legs and body keep it standing
        # the policy acts at the rate its training data was recorded at (every record_every control steps); its last
        # action is held in between
        if step >= settle and (step - settle) % ep["record_every"] == 0:
            mujoco.mj_forward(model, data)  # images from the current pose, as render_episodes draws the training ones
            log["state"].append(data.qpos[joints].astype(np.float32))
            log["qpos"].append(data.qpos.copy())
            log["object"].append(scene.object_pos(model, data))
            images = []
            for cam in ("head", "wrist"):
                eye.update_scene(data, camera=cam)
                images.append(eye.render())
            obs = {"observation.images.head": images[0], "observation.images.wrist": images[1],
                   "observation.state": log["state"][-1]}
            action = predict_action(obs, policy, device, pre, post, use_amp=False, task=TASK,
                                    robot_type="g1_inspire").cpu().numpy().reshape(-1)  # drop the batch dim
            log["action"].append(action)
            if writer:
                eye.update_scene(data, view)
                writer.write(cv2.cvtColor(np.hstack([eye.render(), *images]), cv2.COLOR_RGB2BGR))
        data.ctrl[arm], data.ctrl[fingers] = action[:len(arm)], action[len(arm):]  # overrides the balance policy's arm
        tick = time.perf_counter()
        mujoco.mj_step(model, data, nstep=round(robot.CONTROL_DT / robot.SIM_DT))
        if viewer:
            viewer.sync()
            time.sleep(max(0.0, robot.CONTROL_DT - (time.perf_counter() - tick)))  # real time, to watch
        max_lift = max(max_lift, scene.object_pos(model, data)[2] - start[2])
        if data.qpos[2] < robot.STANDING_HEIGHT / 2:  # pelvis below half its standing height: the robot fell
            fell = True
            break
    if writer:
        writer.release()
    if viewer:
        viewer.close()
    eye.close()
    return {**scene.outcome(model, data, cfg, [*goal_xy, 0.0], max_lift, fell), **{k: np.array(v) for k, v in log.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", type=Path, help="a checkpoint's pretrained_model folder")
    parser.add_argument("--episodes", type=Path, default=Path("data/sample_episodes"), help="where the clips' moves come from")
    parser.add_argument("--trials", type=int, default=5, help="per clip")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid", action="store_true", help="evenly spaced starts (scene.grid_offsets) instead of random")
    parser.add_argument("--videos", type=int, default=1, help="trials per clip saved as video")
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluate"))
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), help="cylinder start [m]; needs --goal")
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"), help="goal [m]")
    parser.add_argument("--random-goals", action="store_true", help="start and goal anywhere in reach (scene.random_placement)")
    parser.add_argument("--live", action="store_true", help="watch in the MuJoCo viewer")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value for this run, e.g. object.mass=0.35 (test scenarios)")
    parser.add_argument("--turn-cylinder", type=float, default=0.0, metavar="DEG",
                        help="turn the lying cylinder about the vertical by this much (stand-up test scenarios)")
    parser.add_argument("--record", type=Path, help="save each successful trial here as a training episode (as replay_demos.py)")
    args = parser.parse_args()
    cfg = config.load()
    for item in args.set:  # e.g. object.mass=0.35 -> cfg["object"]["mass"] = 0.35
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for k in path:
            node = node[k]
        assert last in node, f"--set {key}: no such config value"
        node[last] = ast.literal_eval(value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = get_policy_class(PreTrainedConfig.from_pretrained(args.policy).type).from_pretrained(args.policy)  # ACT or diffusion
    policy.config.device = str(device)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.policy),
                                         preprocessor_overrides={"device_processor": {"device": str(device)}})
    quats = {}  # per clip: the lying cylinder's orientation (stand-up episodes), else upright
    if args.start:  # the policy sees only images and joints, so a placement needs no clip
        placements = {"custom": [(np.array(args.start), np.array(args.goal))]}
    elif args.random_goals:
        rng = np.random.default_rng(args.seed)
        placements = {"random": [scene.random_placement(cfg, rng) for _ in range(args.trials)]}
        model = scene.build_model(cfg)
        object_qpos = model.jnt_qposadr[model.joint("object").id]
        run = next((r for r in sorted(args.episodes.iterdir()) if (r / "data.npz").exists()), None)
        if run is not None and "task" in (d := np.load(run / "data.npz")).files and str(d["task"]) == "stand_up":
            quats["random"] = d["qpos"][0][object_qpos + 3:object_qpos + 7]  # stand-up: the clip's lying cylinder
    else:  # each clip's move, starts drawn per clip as episode.trials does: `replay_demos.py CLIP --trials N --seed S`
        placements = {}  # replays the expert on the same starts, for a like-for-like comparison
        model = scene.build_model(cfg)
        object_qpos = model.jnt_qposadr[model.joint("object").id]
        for run in sorted(args.episodes.iterdir()):
            if (run / "data.npz").exists() and (clip := run.name.rsplit("_", 1)[0]) not in placements:
                d, rng = np.load(run / "data.npz"), np.random.default_rng(args.seed)
                if "task" in d.files and str(d["task"]) == "stand_up":  # the clip's lying orientation, from its episode
                    quats[clip] = d["qpos"][0][object_qpos + 3:object_qpos + 7]
                placements[clip] = [scene.placement(cfg, d["goal"] - d["start"], scene.grid_offsets(cfg, args.trials)[t]
                                                    if args.grid else rng.uniform(-1, 1, 2) * cfg["episode"]["trial_spread"])
                                    for t in range(args.trials)]
    if args.turn_cylinder:  # turn each lying cylinder about the vertical, on top of its recorded orientation
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), np.radians(args.turn_cylinder))
        for clip, q in quats.items():
            turned = np.zeros(4)
            mujoco.mju_mulQuat(turned, turn, q)
            quats[clip] = turned
    args.out.mkdir(parents=True, exist_ok=True)
    total = n = 0
    for clip, trials in placements.items():
        wins = 0
        for trial, (object_xy, goal_xy) in enumerate(trials):
            video = args.out / f"{clip}_{trial}.mp4" if trial < args.videos and not args.live else None
            r = rollout(cfg, policy, device, pre, post, object_xy, goal_xy, video, args.live, quats.get(clip))
            # every trial is saved (poses included), so it can be re-rendered later (scripts/showcase.py)
            np.savez(args.out / f"{clip}_{trial}.npz", start=object_xy, goal=goal_xy,
                     fps=1 / (cfg["episode"]["record_every"] * robot.CONTROL_DT), **r)
            wins += r["success"]
            if args.record and r["success"]:  # self-improvement: the policy's own successes become training data
                run = args.record / f"{clip}_{args.seed}{trial:03d}"
                run.mkdir(parents=True, exist_ok=True)
                names = [*robot.RIGHT_ARM_JOINTS, *robot.RIGHT_HAND_DRIVES]
                np.savez(run / "data.npz", qpos=r["qpos"], state=r["state"], action=r["action"], start=object_xy,
                         goal=goal_xy, fps=1 / (cfg["episode"]["record_every"] * robot.CONTROL_DT), names=names,
                         task="stand_up" if clip in quats else "pick_place")
            print(f"{clip} trial {trial}: lifted {100 * r['lifted']:.1f} cm  goal error {100 * r['goal_error']:.1f} cm"
                  f"  tilt {r['tilt_deg']:.0f} deg{'' if r['released'] else '  NOT RELEASED'}{'  FELL' if r['fell'] else ''}"
                  f"  {'SUCCESS' if r['success'] else 'FAIL'}",
                  flush=True)
        print(f"== {clip}: {wins}/{len(trials)}", flush=True)
        total, n = total + wins, n + len(trials)
    print(f"success rate {total}/{n}")


if __name__ == "__main__":
    main()
