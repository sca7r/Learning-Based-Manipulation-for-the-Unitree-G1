"""Training runs saved by replay_demos.py --record -> one LeRobot dataset (LeRobot format v3),
for training ACT or Diffusion Policy with LeRobot.

  python scripts/to_lerobot.py data/sample_episodes data/lerobot/g1_inspire
  python scripts/to_lerobot.py data/sample_episodes data/episodes_standup data/lerobot/NAME

Each run folder holds data.npz (poses, state, action, joint names, fps, start, goal). The head and wrist camera
images are drawn here from the poses (scripts/render_episodes.frames), so LeRobot's encoding is their only
compression. Needs lerobot and mujoco, with src/ and scripts/ on PYTHONPATH.
"""

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from g1_imitation import config
from render_episodes import frames

# Task text stored with every frame. The policy never reads it (it tells the tasks apart from the images), but
# LeRobot needs one per episode.
TASK = "Pick up the red cylinder and place it upright on the green square."
TASKS = {"pick_place": TASK, "stand_up": "Pick up the lying red cylinder and stand it upright on the green square."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="folders of runs")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    cfg = config.load()
    runs = [d for folder in args.runs for d in sorted(folder.iterdir()) if (d / "data.npz").exists()]
    first = np.load(runs[0] / "data.npz")  # joint names and fps come from the first run; all runs share them
    names = [str(n) for n in first["names"]]
    width, height = cfg["episode"]["record_size"]
    features = {f"observation.images.{cam}": {"dtype": "video", "shape": (height, width, 3),
                                              "names": ["height", "width", "channels"]} for cam in ("head", "wrist")}
    features |= {key: {"dtype": "float32", "shape": (len(names),), "names": names}
                 for key in ("observation.state", "action")}
    dataset = LeRobotDataset.create("local/" + args.out.name, fps=int(first["fps"]), features=features, root=args.out,
                                    robot_type="g1_inspire", vcodec="h264",
                                    streaming_encoding=True)  # encodes as frames arrive, no PNG step; same crf/g
    for run in runs:
        data = np.load(run / "data.npz")
        assert len(data["qpos"]) == len(data["state"]) == len(data["action"]), run  # one image pair per state/action
        for (head, wrist), state, action in zip(frames(cfg, run), data["state"], data["action"]):
            dataset.add_frame({"observation.images.head": head, "observation.images.wrist": wrist,
                               "observation.state": state.astype(np.float32), "action": action.astype(np.float32),
                               "task": TASKS[str(data["task"]) if "task" in data.files else "pick_place"]})  # older runs have no task
        dataset.save_episode()
        print(f"{run.name}: {len(data['state'])} frames")
    dataset.finalize()
    print(f"{len(runs)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
