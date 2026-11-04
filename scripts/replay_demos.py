"""Replay a recorded clip on the robot: the robot uses its own approach and grip, and takes the cylinder's move, the
carry path and the finger timing from the clip (retarget.demo_plan / retarget.standup_plan).

  python scripts/process_demos.py data/raw/IMG_8473.MOV          # once per clip
  python scripts/replay_demos.py IMG_8473                      # one run, writes outputs/replay_demos/<clip>.mp4
  python scripts/replay_demos.py IMG_8473 --live               # watch it in the MuJoCo viewer
  python scripts/replay_demos.py IMG_8473 --trials 20          # randomized cylinder starts, success rate
  python scripts/replay_demos.py IMG_8467 --trials 20 --record data/episodes   # keep the successes as training data

The goal is where the clip put the cylinder relative to where it started (the cylinder track), not a fixed spot.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from g1_imitation import config, episode, retarget, scene
from g1_imitation.robot import REPO_ROOT

# Per stand-up clip, the simulator's grasp correction (retarget.standup_plan; written by scripts/standup_grasp_search.py)
STANDUP_GRASP = REPO_ROOT / "data" / "standup_grasp.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="clip name, e.g. IMG_8473")
    parser.add_argument("--config", type=Path, default=config.DEFAULT_PATH)
    parser.add_argument("--live", action="store_true", help="watch each run in the MuJoCo viewer instead of saving its video")
    parser.add_argument("--trials", type=int, default=1, help="randomize the object start and report the success rate")
    parser.add_argument("--record", type=Path, help="save each successful trial here as training data")
    parser.add_argument("--seed", type=int, default=0, help="random object starts")
    parser.add_argument("--grid", action="store_true", help="evenly spaced starts (scene.grid_offsets) instead of random")
    parser.add_argument("--random-goals", action="store_true", help="start and goal both random in the workspace")
    args = parser.parse_args()
    cfg = config.load(args.config)

    processed = REPO_ROOT / "data" / "processed" / f"{args.clip}.npz"
    phases = retarget.read_phases().get(args.clip)
    if not processed.exists() or phases is None:
        raise SystemExit(f"{args.clip} isn't processed yet: run python scripts/process_demos.py data/raw/{args.clip}.MOV")
    demo = np.load(processed)
    # the task comes from data/sessions.csv; clips missing there are treated as pick-and-place
    task = next((r["task"] for r in csv.DictReader(open(REPO_ROOT / "data" / "sessions.csv")) if r["clip"] == args.clip),
                "pick_place")
    offsets = scene.grid_offsets(cfg, args.trials) if args.grid else None
    video = REPO_ROOT / "outputs" / "replay_demos" / f"{args.clip}.mp4"
    if task == "stand_up":  # lying cylinder: the clip's own grasp, closing and wrist turn (retarget.standup_plan)
        fix = {k: float(v) for k, v in next(r for r in csv.DictReader(open(STANDUP_GRASP)) if r["clip"] == args.clip).items()
               if k != "clip"}
        # stand-up grips with the fingertips only: the palm must stay off the cylinder, and the finger forces differ
        cfg["grasp"]["palm_free"] = cfg["standup"]["palm_free"]
        cfg["grasp"]["finger_force_limits"] = cfg["standup"]["finger_force_limits"]
        make_plan, move, quat = retarget.standup_plan(cfg, demo, phases, fix)
        episode.trials(cfg, make_plan, args.trials, video, args.live, goal_offset=move, record_dir=args.record,
                       seed=args.seed, offsets=offsets, object_quat=quat, task=task)
        return
    # pick-and-place: a new plan per trial, because the start (and with --random-goals the goal) changes each time
    episode.trials(cfg, lambda start, goal: retarget.demo_plan(cfg, demo, phases, start, goal, rotate=args.random_goals), args.trials,
                   video, args.live, goal_offset=retarget.demo_move(demo, phases), record_dir=args.record, seed=args.seed,
                   offsets=offsets, random_goals=args.random_goals)


if __name__ == "__main__":
    main()
