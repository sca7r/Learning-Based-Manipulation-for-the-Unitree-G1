"""Stand-up replay with the grasp settings given in the environment (for scripts/standup_grasp_search.py), set up as
replay_demos.py runs a stand_up clip. Prints each trial's result and where the cylinder landed relative to the goal.

  ALONG=0 ACROSS=0 DEEPER=0 YAW=0 ROLL=0 PITCH=0 CLOSE=1 THUMB=1.2,0.2 [PLACE=dx,dy] [SEED=0] [GRID=1] \
      PYTHONPATH=src python scripts/standup_grasp_check.py CLIP OUT_DIR N_TRIALS

The first VIDEOS trials (default 2) are saved as video in OUT_DIR.
"""
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

from g1_imitation import config, episode, retarget, scene
from g1_imitation.robot import REPO_ROOT

clip, out, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
cfg = config.load()
cfg["grasp"]["palm_free"] = cfg["standup"]["palm_free"]  # as replay_demos.py for stand_up clips
cfg["grasp"]["finger_force_limits"] = cfg["standup"]["finger_force_limits"]
thumb = [float(v) for v in os.environ["THUMB"].split(",")]
place = [float(v) for v in os.environ.get("PLACE", "0,0").split(",")]
# the grasp corrections in data/standup_grasp.csv's column names, read from the environment (unset = 0)
fix = {k: float(os.environ.get(k.upper(), 0)) for k in ("along", "across", "deeper", "yaw", "roll", "pitch")} | dict(
    close=float(os.environ["CLOSE"]), thumb_yaw=thumb[0], thumb_pitch=thumb[1], place_x=place[0], place_y=place[1])
make_plan, move, quat = retarget.standup_plan(cfg, np.load(REPO_ROOT / "data" / "processed" / f"{clip}.npz"),
                                              retarget.read_phases()[clip], fix)
z = cfg["scene"]["table_height"] + cfg["object"]["height"] / 2
rng, wins = np.random.default_rng(int(os.environ.get("SEED", 0))), 0
for i in range(n):
    # where the cylinder starts, relative to the clip's own start: evenly spread (GRID=1) or random
    off = scene.grid_offsets(cfg, n)[i] if os.environ.get("GRID") else rng.uniform(-1, 1, 2) * cfg["episode"]["trial_spread"]
    start, goal = scene.placement(cfg, move, off)
    plan, duration, events = make_plan(np.r_[start, z], np.r_[goal, z])
    final = []
    step = mujoco.mj_step

    def logged(m, d, nstep=1):  # remember the cylinder's last position
        step(m, d, nstep=nstep)
        final[:] = [d.xpos[m.body("object").id][:2].copy()]

    # episode.run doesn't return where the cylinder ended up, so wrap its physics step for this run to record it
    episode.mujoco.mj_step = logged
    r = episode.run(cfg, plan, duration, events, start, video_path=out / f"{clip}_{i}.mp4" if i < int(os.environ.get("VIDEOS", 2)) else None,
                    goal_xy=goal, object_quat=quat)
    episode.mujoco.mj_step = step  # put the real step back
    wins += r["success"]
    dx, dy = 100 * (final[0] - goal) if final else (np.nan, np.nan)
    print(f"{episode.report(i, r)}  landed {dx:+.1f} {dy:+.1f} cm", flush=True)
print(f"{clip} success {wins}/{n}")
