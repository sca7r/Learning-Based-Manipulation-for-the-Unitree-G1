"""Simulator grasp search for one stand-up clip (Human2Sim2Robot / VideoManip style: correct the video's grasp in the
simulator): Optuna CMA-ES over the corrections in data/standup_grasp.csv (along, across, deeper [m], yaw, roll, pitch
[deg], close, thumb_yaw, thumb_pitch), each candidate run on TRIALS random starts with scripts/standup_grasp_check.py.
Score per trial: lift (up to 10 cm) + how upright it ends once lifted 5 cm; 0 if the palm touched. Where it lands is
not scored: the per-clip set-down correction (place_x, place_y) is measured afterwards. Starts from the clip's row in
data/standup_grasp.csv if it has one.

  TRIALS=3 JOBS=5 PYTHONPATH=src python scripts/standup_grasp_search.py CLIP CANDIDATES
"""
import csv
import os
import re
import subprocess
import sys

import optuna

from g1_imitation.robot import REPO_ROOT

clip, budget, trials = sys.argv[1], int(sys.argv[2]), int(os.environ.get("TRIALS", 3))
out = REPO_ROOT / "outputs" / "standup_grasp_search" / clip
out.mkdir(parents=True, exist_ok=True)


def score_line(line: str) -> float:
    """Score of one trial line printed by standup_grasp_check.py, from 0 to 2: half for the lift (capped at 10 cm),
    half for ending upright, which only counts once the cylinder actually came up 5 cm."""
    m = re.search(r"lifted +([\d.]+) cm +goal error +([\d.]+) cm +tilt +([\d.]+) deg", line)
    if not m or "palm" in line:
        return 0.0
    lift, tilt = float(m[1]), float(m[3])
    return min(lift, 10) / 10 + (1 - min(tilt, 90) / 90) * (lift >= 5)


def objective(t: optuna.Trial) -> float:
    """Run one candidate grasp in its own process (the settings go in through the environment) and return its mean
    score. Every candidate gets the same starts (SEED=7), so candidates are compared on equal terms."""
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": "src", "OMP_NUM_THREADS": "1", "SEED": "7",
           "ALONG": str(t.suggest_float("along", -0.05, 0.05)), "ACROSS": str(t.suggest_float("across", -0.03, 0.03)),
           "DEEPER": str(t.suggest_float("deeper", -0.02, 0.04)), "YAW": str(t.suggest_float("yaw", -40, 40)),
           "ROLL": str(t.suggest_float("roll", -20, 20)), "PITCH": str(t.suggest_float("pitch", -20, 20)),
           "CLOSE": str(t.suggest_float("close", 1.0, 3.0)),
           "THUMB": f"{t.suggest_float('thumb_yaw', 0.0, 1.3)},{t.suggest_float('thumb_pitch', 0.0, 0.6)}"}
    r = subprocess.run([sys.executable, "scripts/standup_grasp_check.py", clip, str(out / f"t{t.number}"), str(trials)],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("trial")]
    # also count trials that would pass the real test's lift and tilt part, just for the log
    s = sum(map(score_line, lines)) / trials
    up = sum(bool(m) and float(m[1]) >= 8 and float(m[2]) <= 15 and "palm" not in l for l in lines
             for m in [re.search(r"lifted +([\d.]+) cm .* tilt +([\d.]+) deg", l)])
    print(f"{clip} candidate {t.number}: score {s:.2f}  lifted and upright {up}/{trials}"
          f"  {t.params}", flush=True)
    return s


# CMA-ES suits this: 9 continuous parameters and a noisy score. popsize 10 lets JOBS candidates run in parallel.
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.CmaEsSampler(seed=0, popsize=10))
row = next((r for r in csv.DictReader(open(REPO_ROOT / "data" / "standup_grasp.csv")) if r["clip"] == clip), None)
if row:  # start from the clip's current settings, so the search can only improve on them
    study.enqueue_trial({k: float(v) for k, v in row.items() if k not in ("clip", "place_x", "place_y")})
study.optimize(objective, n_trials=budget, n_jobs=int(os.environ.get("JOBS", 3)))
print(f"{clip} best score {study.best_value:.2f} {study.best_params}")
