"""Draws the head and wrist camera videos of recorded training episodes from their saved poses.

  python scripts/render_episodes.py data/sample_episodes
  python scripts/render_episodes.py data/sample_episodes --shard 0/4   # and 1/4, 2/4, 3/4 at the same time

Generation (replay_demos.py --record) saves only data.npz with the simulation pose (qpos) at each recorded frame; this
rebuilds the same scene (start and goal from data.npz), sets each pose and renders [scene] cameras at
[episode] record_size into head.mp4 and wrist.mp4, the same images run() would have rendered at that step.
Rendering is split off because under WSL MuJoCo's OpenGL goes through a slow translation layer (~10 images/s on an
RTX 3060), while native Windows renders ~650/s; physics and planning (OMPL) stay in WSL. Skips episodes already done.
These MP4s are for watching; scripts/to_lerobot.py draws the training images itself with frames(), so they are
compressed once (by LeRobot), not twice.
"""

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from g1_imitation import config, scene


def frames(cfg: dict, episode: Path):
    """(head, wrist) RGB images for each recorded frame of an episode, drawn from its saved poses."""
    rec = np.load(episode / "data.npz")
    model = scene.build_model(cfg, rec["start"], rec["goal"])
    data = mujoco.MjData(model)
    width, height = cfg["episode"]["record_size"]
    renderer = mujoco.Renderer(model, height, width)
    for qpos in rec["qpos"]:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)  # positions only, no physics step: the pose is exactly the recorded one
        images = []
        for cam in ("head", "wrist"):
            renderer.update_scene(data, camera=cam)
            images.append(renderer.render())
        yield images
    renderer.close()


def render(cfg: dict, episode: Path) -> int:
    """Write head.mp4 and wrist.mp4 into the episode folder; returns the number of frames."""
    width, height = cfg["episode"]["record_size"]
    fps = float(np.load(episode / "data.npz")["fps"])
    writers = [cv2.VideoWriter(str(episode / f"{cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
               for cam in ("head", "wrist")]
    n = 0
    for images in frames(cfg, episode):
        for writer, image in zip(writers, images):
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        n += 1
    for writer in writers:
        writer.release()
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="folder of episodes, each <name>/data.npz")
    parser.add_argument("--config", type=Path, default=config.DEFAULT_PATH)
    parser.add_argument("--shard", default="0/1", help="i/n: render every n-th episode from i, to run n in parallel")
    args = parser.parse_args()
    cfg = config.load(args.config)
    i, n = map(int, args.shard.split("/"))
    # skip episodes that already have their videos, so an interrupted run can just be started again
    todo = sorted(e for e in args.root.iterdir() if (e / "data.npz").exists() and not (e / "wrist.mp4").exists())[i::n]
    for i, episode in enumerate(todo):
        print(f"{i + 1}/{len(todo)} {episode.name}: {render(cfg, episode)} frames", flush=True)


if __name__ == "__main__":
    main()
