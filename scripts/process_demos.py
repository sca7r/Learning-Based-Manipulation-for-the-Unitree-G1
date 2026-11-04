"""Turn recorded clips into demonstrations: hand keypoints and orientation on the table, Inspire finger angles, the
cylinder's position, grasp phases.

  python scripts/process_demos.py data/raw/IMG_*.MOV

Hands come from data/hands/<clip>.npz: run scripts/track_hands_wilor.py on a GPU machine first.

For each clip:
  data/processed/<clip>.npz   time [s], keypoints (frames x 21 x 3 in the table frame [m], NaN = no hand),
                              hand_rotation (frames x 3 x 3, columns: toward the wrist, back-of-hand normal, index
                              side; retarget.hand_frame), finger angles (frames x 6, Inspire drive joints),
                              cylinder_xy (frames x 2, where the cylinder stands on the table, NaN = not fully
                              visible, e.g. in the hand), camera pose
  data/phases.csv             first frame of each phase (reach, grasp, carry, place, retract), one row per clip
  outputs/demos/<clip>.mp4    the clip with the hand skeleton and axes, the cylinder's outline where it was found,
                              table axes, frame number, phase and wrist height

Phases are labelled automatically from finger closure and wrist height. Check them in the overlay video and fix
any wrong frame numbers in data/phases.csv: rows already in the file are kept as they are, so rerunning redraws
the overlay with your corrections.
Keypoint order is OpenPose's: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from g1_imitation import config, hand_tracking, retarget
from g1_imitation.robot import REPO_ROOT

# keypoint chains drawn in the overlay video: one line per finger, plus the knuckle line
SKELETON = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [0, 17, 18, 19, 20], [5, 9, 13, 17]]


def frames(path: Path):
    """The clip's frames one by one, so a 4K clip never has to fit in memory."""
    video = cv2.VideoCapture(str(path))
    while True:
        ok, frame = video.read()
        if not ok:
            return
        yield frame


def process(clip: Path, cfg: dict, phases: dict, sheet_turn: float | None = None) -> None:
    """Writes data/processed/<clip>.npz and the overlay video, and adds the clip's phases to `phases` if missing."""
    video = cv2.VideoCapture(str(clip))
    if not video.isOpened():
        raise SystemExit(f"{clip}: can't open the video (no such file, or a format OpenCV can't read)")
    fps = video.get(cv2.CAP_PROP_FPS)
    width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    # The camera doesn't move during a clip: fit its field of view to the markers, then take the sheet pose as the
    # median over every frame showing them.
    views = [v for f in frames(clip) if (v := hand_tracking.marker_points(f, cfg["hand_tracking"])) is not None]
    if not views:
        raise SystemExit(f"{clip}: none of the table markers ([hand_tracking] markers) found in any frame")
    K = hand_tracking.fit_camera(views[::max(1, len(views) // 20)], width, height)
    poses = [cv2.solvePnP(points, pixels, K, None, flags=cv2.SOLVEPNP_IPPE)[1:] for points, pixels in views]
    R = cv2.Rodrigues(np.median([r.ravel() for r, _ in poses], axis=0))[0]
    t = np.median([t.ravel() for _, t in poses], axis=0)

    # Hands: WiLoR's right hand, from scripts/track_hands_wilor.py (run on a GPU machine).
    wilor = REPO_ROOT / "data" / "hands" / f"{clip.stem}.npz"
    if not wilor.exists():
        raise SystemExit(f"{wilor} missing: run scripts/track_hands_wilor.py on {clip.name} first (needs a GPU)")
    radius, cylinder_height = cfg["object"]["radius"], cfg["object"]["height"]
    cylinder, cylinder_yaw = [], []
    for i, frame in enumerate(frames(clip)):
        # ponytail: ~20 checks per second are plenty for an object that rests or moves slowly; it is the slowest step
        found = None if i % max(1, round(fps / 20)) else hand_tracking.find_cylinder(
            frame, R, t, K, radius, cylinder_height, cfg["hand_tracking"]["cylinder"])
        cylinder.append(np.full(2, np.nan) if found is None else found[0])
        cylinder_yaw.append(np.nan if found is None or found[1] is None else found[1])  # nan: upright or not seen
    cylinder, cylinder_yaw = np.array(cylinder), np.array(cylinder_yaw)
    w = np.load(wilor)
    if len(w["pixels"]) != len(cylinder):
        raise SystemExit(f"{wilor}: {len(w['pixels'])} frames, the clip has {len(cylinder)}; rerun track_hands_wilor.py")
    hands = [None if np.isnan(p[0, 0]) else (p, s) for p, s in zip(w["pixels"], w["shape"])]
    pixels = [h and h[0] for h in hands]
    in_camera = np.array([hand_tracking.hand_in_camera(*h, K) if h else np.full((21, 3), np.nan) for h in hands])
    keypoints = (in_camera - t) @ R  # camera -> table
    tracked = ~np.isnan(keypoints[:, 0, 0])
    if not tracked.any():
        raise SystemExit(f"{clip}: no hand found")
    # Sheet frame -> table frame, from the hand resting at the start (its first second of tracking).
    turn = hand_tracking.sheet_to_table(keypoints[np.flatnonzero(tracked)[:round(fps)]])
    if sheet_turn is not None:  # the resting hand wasn't pointing away from the demonstrator
        a = np.radians(sheet_turn)
        turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    keypoints, cylinder, R = keypoints @ turn.T, cylinder @ turn[:2, :2].T, R @ turn.T
    cylinder_yaw = (cylinder_yaw + np.arctan2(turn[1, 0], turn[0, 0])) % np.pi  # a lying axis has no front: 0..pi
    rotations = np.array([retarget.hand_frame(k) if ok else np.full((3, 3), np.nan) for k, ok in zip(keypoints, tracked)])

    names, fingers = retarget.finger_angles(keypoints)
    closed = retarget.closure(names, fingers)
    wrist_height = np.interp(np.arange(len(keypoints)), np.flatnonzero(tracked), keypoints[tracked, 0, 2])  # gaps filled
    if clip.stem not in phases:  # rows already in phases.csv may be hand-corrected, so they are never relabelled
        try:
            phases[clip.stem] = retarget.label_phases(wrist_height, closed, cfg["demos"])
        except ValueError as error:
            print(f"{clip.name}: could not label phases ({error}); add its row to {retarget.PHASES_CSV.name} by hand")
    starts = phases.get(clip.stem, {})

    out = REPO_ROOT / "data" / "processed" / f"{clip.stem}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, time=np.arange(len(keypoints)) / fps, keypoints=keypoints, hand_rotation=rotations, finger_joints=names,
             fingers=fingers, cylinder_xy=cylinder, cylinder_yaw=cylinder_yaw, camera_from_table_R=R, camera_from_table_t=t, K=K)

    overlay_path = REPO_ROOT / "outputs" / "demos" / f"{clip.stem}.mp4"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i, frame in enumerate(frames(clip)):
        if pixels[i] is not None:
            for chain in SKELETON:
                cv2.polylines(frame, [pixels[i][chain].astype(np.int32)], False, (0, 255, 0), 2)
            cv2.drawFrameAxes(frame, K, None, cv2.Rodrigues(R @ rotations[i])[0], R @ keypoints[i, 0] + t, 0.05)
        if not np.isnan(cylinder[i, 0]):
            outline = hand_tracking.cylinder_outline(cylinder[i], R, t, K, radius, cylinder_height)
            cv2.polylines(frame, [outline], True, (0, 255, 255), 2)
        cv2.drawFrameAxes(frame, K, None, cv2.Rodrigues(R)[0], t, cfg["hand_tracking"]["marker_size"])
        phase = max((p for p in starts if starts[p] <= i), key=starts.get, default="?")  # the latest phase begun
        text = f"frame {i}  {phase}  wrist {100 * wrist_height[i]:.1f} cm  closed {100 * closed[i]:.0f}%"
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        overlay.write(frame)
    overlay.release()
    seen = np.flatnonzero(~np.isnan(cylinder[:, 0]))
    where = (f"from {np.round(100 * cylinder[seen[0]]).astype(int)} to {np.round(100 * cylinder[seen[-1]]).astype(int)} cm"
             if len(seen) else "")
    print(f"{clip.name}: cylinder in {len(seen)}/{len(cylinder)} frames {where}")
    print(f"{clip.name}: hand (WiLoR) in {tracked.sum()}/{len(tracked)} frames, marker in {len(poses)}, "
          f"camera {100 * np.linalg.norm(t):.0f} cm from the marker, field of view "
          f"{np.degrees(2 * np.arctan(width / 2 / K[0, 0])):.1f} deg, sheet turned "
          f"{np.degrees(np.arctan2(turn[1, 0], turn[0, 0])):.0f} deg, phases {starts or 'missing'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=config.DEFAULT_PATH)
    parser.add_argument("--sheet-turn", type=float, help="sheet direction [deg] when the resting hand misleads (printed per clip)")
    args = parser.parse_args()
    cfg = config.load(args.config)
    phases = retarget.read_phases()
    for clip in args.clips:
        process(clip, cfg, phases, args.sheet_turn)
    retarget.write_phases(phases)


if __name__ == "__main__":
    main()
