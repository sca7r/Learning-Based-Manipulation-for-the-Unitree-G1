"""A small learned correction for the stand-up expert: while the hand holds the cylinder, a linear policy adds a little
rotation and shift to the hand target so the cylinder ends upright on the goal, even when it pivots in the thumb-index
pinch. Trained with Augmented Random Search (Mania, Guy, Recht 2018, arXiv 1803.07055: linear policies, observation
normalisation, +/- random directions, top-b directions, step scaled by the reward spread). Every direction is scored
on the same starts of every clip in an iteration (common random numbers), so a start's difficulty doesn't decide it.

Observation (8, normalised): the cylinder axis minus where it would be had it turned rigidly with the hand since the
grip (the pivot in the grip, 3), the cylinder's angular velocity (3), time since the grip / 10 s (1), constant (1).
Action (6): hand rotation (axis-angle, up to MAX_TURN rad) and shift (up to MAX_SHIFT m), added to the plan's target.
Simulator-only inputs: this improves the expert that generates data; the trained policy (ACT) sees only images.

  PYTHONPATH=src python scripts/standup_rl.py train IMG_8489 IMG_8510 --iterations 100
  PYTHONPATH=src python scripts/standup_rl.py check IMG_8489 IMG_8510 --trials 10     # 10 new starts, every video
Grasp settings per clip: data/standup_grasp.csv. Policy: outputs/standup_rl/policy.npz.
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from g1_imitation import config, episode, retarget, scene
from g1_imitation.robot import REPO_ROOT

MAX_TURN, MAX_SHIFT = 0.26, 0.03  # rad (15 deg), m
OBS, ACT = 8, 6  # observation and action sizes (see the module docstring)
OUT = REPO_ROOT / "outputs" / "standup_rl"


def setup(clip: str):
    """(cfg, make_plan, move, quat) for a stand-up clip, as replay_demos.py builds them."""
    cfg = config.load()
    cfg["grasp"]["palm_free"] = cfg["standup"]["palm_free"]
    cfg["grasp"]["finger_force_limits"] = cfg["standup"]["finger_force_limits"]
    fix = {k: float(v) for k, v in next(r for r in csv.DictReader(open(REPO_ROOT / "data" / "standup_grasp.csv"))
                                        if r["clip"] == clip).items() if k != "clip"}
    make_plan, move, quat = retarget.standup_plan(cfg, np.load(REPO_ROOT / "data" / "processed" / f"{clip}.npz"),
                                                  retarget.read_phases()[clip], fix)
    return cfg, make_plan, move, quat


def rotation(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of an axis-angle vector (direction = axis, length = angle in rad)."""
    m = np.zeros(9)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, v / max(np.linalg.norm(v), 1e-12), np.linalg.norm(v))
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def rollout(clip: str, W: np.ndarray, mean: np.ndarray, std: np.ndarray, seed: int, video: Path | None = None):
    """One stand-up try with policy W at a random start. Returns (reward, success, observations, result)."""
    cfg, make_plan, move, quat = setup(clip)
    rng = np.random.default_rng(seed)
    start, goal = scene.placement(cfg, move, rng.uniform(-1, 1, 2) * cfg["episode"]["trial_spread"])
    z = cfg["scene"]["table_height"] + cfg["object"]["height"] / 2
    plan, duration, events = make_plan(np.r_[start, z], np.r_[goal, z])
    seen, held = [], {}

    def feedback(model, data, clock, position, rotation_):
        """Called by episode.run every control step with the planned hand target; returns the corrected target.
        Does nothing until the hand has a firm grip."""
        b, site = model.body("object").id, data.site("right_grasp")
        if clock < events["grip"] or not scene.firm_grip(scene.touching(model, data)):
            return position, rotation_
        axis = data.xmat[b].reshape(3, 3)[:, 2]
        if not held:  # the cylinder's axis in the hand frame at the grip
            held.update(axis=site.xmat.reshape(3, 3).T @ axis, t=clock)
        rigid = rotation_ @ held["axis"]  # where the axis would point if the cylinder had not moved in the grip
        rigid *= np.sign(rigid @ axis) or 1.0  # the axis has no front
        obs = np.r_[axis - rigid, data.cvel[b][:3], (clock - held["t"]) / 10, 1.0]
        seen.append(obs)
        a = np.clip(W @ ((obs - mean) / std), -1, 1)  # linear policy on normalised observations
        return position + MAX_SHIFT * a[3:], rotation(MAX_TURN * a[:3]) @ rotation_

    r = episode.run(cfg, plan, duration, events, start, video_path=video, goal_xy=goal, object_quat=quat,
                    feedback=feedback)
    # reward: lift (capped at 10 cm), then upright, then close to the goal, then a bonus for full success; touching
    # the cylinder with the palm scores 0, as in the grasp search
    lifted = r["lifted"] >= 0.05
    reward = 0.0 if r["error"] and "palm" in r["error"] else (
        min(r["lifted"], 0.1) / 0.1 + lifted * (1 - min(r["tilt_deg"], 90) / 90)
        + lifted * (1 - min(r["goal_error"], 0.2) / 0.2) + r["success"])
    return reward, bool(r["success"]), np.array(seen).reshape(-1, OBS), r


def _job(args):  # Pool.map passes one argument; results without the (large) episode record
    return rollout(*args)[:3]


def train(clips: list[str], iterations: int, directions: int, top: int, step: float, noise: float, workers: int,
          starts: int):
    """ARS: each iteration tries `directions` random changes of W (each + and -), keeps the `top` best and steps W
    toward them. Saves the best policy so far to outputs/standup_rl/policy.npz."""
    OUT.mkdir(parents=True, exist_ok=True)
    W = np.zeros((ACT, OBS))  # zero = the expert as it is
    mean, var, count = np.zeros(OBS), np.ones(OBS), 1e-4
    best = -np.inf
    with Pool(workers) as pool:
        for it in range(iterations):
            std = np.sqrt(np.maximum(var, 1e-6))
            std[-1] = 1.0  # the constant input stays 1
            mean_used = mean.copy()
            mean_used[-1] = 0.0
            deltas = np.random.default_rng(it).standard_normal((directions, ACT, OBS))
            # common random numbers: every direction, + and -, on the same starts of every clip this iteration
            jobs = [(clip, W + sign * noise * deltas[k], mean_used, std, 10_000 * it + s)
                    for k in range(directions) for sign in (1, -1) for clip in clips for s in range(starts)]
            out = pool.map(_job, jobs)
            rewards = np.array([o[0] for o in out]).reshape(directions, 2, len(clips) * starts).mean(axis=2)
            for _, _, obs in out:  # running observation statistics (ARS V2)
                if len(obs):
                    n = len(obs)
                    delta = obs.mean(0) - mean
                    mean = mean + delta * n / (count + n)
                    var = (var * count + obs.var(0) * n + delta ** 2 * count * n / (count + n)) / (count + n)
                    count += n
            order = np.argsort(-rewards.max(axis=1))[:top]  # directions whose better side scored highest
            spread = rewards[order].std() + 1e-6  # step size scaled by the reward spread, as in the paper
            W = W + step / (top * spread) * np.einsum("k,kij->ij", rewards[order, 0] - rewards[order, 1], deltas[order])
            successes = sum(o[1] for o in out)
            print(f"iteration {it}: mean reward {rewards.mean():.2f}, successes {successes}/{len(out)}", flush=True)
            if rewards.mean() > best:
                best = rewards.mean()
                np.savez(OUT / "policy.npz", W=W, mean=mean_used, std=std, iteration=it, reward=best)


def check(clips: list[str], trials: int):
    """Run the saved policy on new starts of each clip, with a video of every trial."""
    p = np.load(OUT / "policy.npz")
    print(f"policy from iteration {int(p['iteration'])} (mean reward {float(p['reward']):.2f})")
    for clip in clips:
        wins = 0
        for t in range(trials):
            reward, success, _, r = rollout(clip, p["W"], p["mean"], p["std"], seed=1_000_000 + t,  # unseen starts
                                            video=OUT / "check" / f"{clip}_{t}.mp4")
            wins += success
            print(f"{clip} {episode.report(t, r)}", flush=True)
        print(f"== {clip}: {wins}/{trials}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["train", "check"])
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--directions", type=int, default=14)
    parser.add_argument("--top", type=int, default=7)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--starts", type=int, default=3, help="starts per clip, shared by all directions in an iteration")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.clips, args.iterations, args.directions, args.top, args.step, args.noise, args.workers, args.starts)
    else:
        check(args.clips, args.trials)


if __name__ == "__main__":
    main()
