"""Human demonstration -> robot: Inspire finger angles (dex-retargeting), grasp phases, and the hand path the
robot's right hand follows.
"""

import csv
from pathlib import Path

import dex_retargeting
import mujoco
import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig

from g1_imitation import episode, robot, scene

# dex-retargeting's own Inspire config: vector retargeting of wrist-to-fingertip vectors from the 21 hand keypoints.
RETARGET_CONFIG = Path(dex_retargeting.__file__).parent / "configs" / "teleop" / "inspire_hand_right.yml"
URDF_DIR = robot.REPO_ROOT / "assets" / "dex_urdf"
# Hand frame -> MANO convention for a right hand, from dex-retargeting's example/vector_retargeting/single_hand_detector.py
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
PHASES = ("reach", "grasp", "carry", "place", "retract")
PHASES_CSV = robot.REPO_ROOT / "data" / "phases.csv"
# Demo table frame (ArUco marker: x to the demonstrator's right, y away from them, z up) -> robot world
# (x forward, y left, z up). Holds when the marker sheet is taped so its text reads normally from the seat.
TABLE_TO_ROBOT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
# hand_frame's axes (toward the wrist, back-of-hand normal, index side) -> the robot hand's grasp frame
# (along the fingers, palm normal, index side).
HAND_FRAME_TO_GRASP = np.diag([-1.0, -1.0, 1.0])


def read_phases() -> dict[str, dict[str, int]]:
    if not PHASES_CSV.exists():
        return {}
    with open(PHASES_CSV) as f:
        return {row["clip"]: {p: int(row[p]) for p in PHASES} for row in csv.DictReader(f)}


def write_phases(phases: dict[str, dict[str, int]]) -> None:
    with open(PHASES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", *PHASES])
        writer.writeheader()
        for clip in sorted(phases):
            writer.writerow({"clip": clip, **phases[clip]})


def _nearest(tracked: np.ndarray, n: int) -> np.ndarray:
    """For each of n frames, the index into tracked of the nearest tracked frame."""
    return np.abs(np.arange(n)[:, None] - tracked[None, :]).argmin(axis=1)


def _smooth(x: np.ndarray, frames: int) -> np.ndarray:
    """Centered moving average along axis 0, edges padded with their own value."""
    pad = frames // 2
    padded = np.concatenate([np.repeat(x[:1], pad, axis=0), x, np.repeat(x[-1:], frames - 1 - pad, axis=0)])
    return np.stack([padded[i:i + len(x)] for i in range(frames)]).mean(axis=0)


def hand_frame(points: np.ndarray) -> np.ndarray:
    """Wrist orientation from the wrist, index base and middle base keypoints (dex-retargeting's convention)."""
    palm = points[[0, 5, 9]]
    wrist, index, middle = palm
    x = wrist - middle
    normal = np.linalg.svd(palm - palm.mean(axis=0))[2][2]  # normal of the palm plane
    x = x - (x @ normal) * normal
    x /= np.linalg.norm(x)
    z = np.cross(x, normal)
    if z @ (index - middle) < 0:
        normal, z = -normal, -z
    return np.stack([x, normal, z], axis=1)


def finger_angles(keypoints: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Inspire drive joint names (our model's) and their angles for each frame. keypoints: frames x 21 x 3, NaN = no hand.

    Frames without a hand repeat the nearest tracked frame.
    """
    RetargetingConfig.set_default_urdf_dir(str(URDF_DIR))
    # The config's low-pass filter is meant for live teleop; offline it would only delay the signal.
    retargeting = RetargetingConfig.load_from_file(RETARGET_CONFIG, override={"low_pass_alpha": 1}).build()
    optimizer = retargeting.optimizer
    origin, task = optimizer.target_link_human_indices
    columns = [optimizer.robot.dof_joint_names.index(n) for n in optimizer.target_joint_names]

    tracked = np.flatnonzero(~np.isnan(keypoints[:, 0, 0]))
    angles = np.zeros((len(tracked), len(columns)))
    for row, i in enumerate(tracked):
        points = keypoints[i] - keypoints[i, 0]
        local = points @ hand_frame(points) @ OPERATOR2MANO_RIGHT
        angles[row] = retargeting.retarget(local[task] - local[origin])[columns]
    return ["R_" + n for n in optimizer.target_joint_names], angles[_nearest(tracked, len(keypoints))]


def closure(names: list[str], angles: np.ndarray) -> np.ndarray:
    """How closed the four fingers are, 0 = most open in this clip, 1 = most closed."""
    curl = angles[:, [i for i, n in enumerate(names) if "thumb" not in n]].mean(axis=1)
    return (curl - curl.min()) / max(curl.max() - curl.min(), 1e-6)


def label_phases(wrist_height: np.ndarray, closed: np.ndarray, cfg: dict) -> dict[str, int]:
    """First frame of each phase, from finger closure (0..1) and wrist height above the table [m].

    grasp: fingers close. carry: wrist rises above its grasp height. place: after its highest point, the wrist is
    back down near its release height. retract: fingers open. Raises ValueError if the clip doesn't contain a grasp and a release.
    The grasp is the longest run of closed fingers: a hand resting flat on the table, or seen from behind, can also
    read as closed for a while, but never as long as the hold.
    """
    is_closed = _smooth(closed, cfg["smoothing_frames"]) > cfg["closed_fraction"]
    if not is_closed.any():
        raise ValueError("fingers never close")
    edges = np.flatnonzero(np.diff(np.r_[0, is_closed.astype(int), 0]))  # run starts and ends, alternating
    starts, ends = edges[::2], edges[1::2]
    longest = int(np.argmax(ends - starts))
    grasp, retract = int(starts[longest]), int(ends[longest])
    if retract == len(is_closed):
        raise ValueError("fingers never open again after the grasp")
    raised = wrist_height[grasp:retract] > wrist_height[grasp] + cfg["lift_height"]
    if not raised.any():
        raise ValueError("wrist never rises between grasp and release")
    carry = grasp + int(np.argmax(raised))
    peak = carry + int(np.argmax(wrist_height[carry:retract]))
    place = peak + int(np.argmax(wrist_height[peak:retract + 1] <= wrist_height[retract] + cfg["lift_height"]))
    return dict(zip(PHASES, (0, grasp, carry, place, retract)))


def hand_poses(keypoints: np.ndarray, object_radius: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, the human hand's grasp point and grasp frame, in the table frame and the robot hand's axes.

    The grasp point is the robot's right_grasp site on a human hand: the finger roots, moved off the palm by half
    the palm's thickness plus one object radius. Frames without a hand are interpolated, then everything is smoothed.
    """
    tracked = np.flatnonzero(~np.isnan(keypoints[:, 0, 0]))
    rotations = np.array([hand_frame(keypoints[i]) @ HAND_FRAME_TO_GRASP for i in tracked])
    roots = keypoints[tracked][:, [5, 9, 13, 17]].mean(axis=1)
    points = roots + rotations[:, :, 1] * (cfg["palm_to_joints"] + object_radius)

    frames = np.arange(len(keypoints))
    points = np.stack([np.interp(frames, tracked, points[:, k]) for k in range(3)], axis=1)
    rotations = _smooth(rotations[_nearest(tracked, len(keypoints))], cfg["smoothing_frames"])
    u, _, vt = np.linalg.svd(rotations)  # back to the nearest rotation matrices after averaging
    return _smooth(points, cfg["smoothing_frames"]), u @ vt


def demo_move(demo, phases: dict[str, int]) -> np.ndarray:
    """Where the demonstration put the cylinder relative to where it stood, in the robot's table plane [m]: medians of
    the cylinder track before the grasp and after the release (the hand covers it in between)."""
    xy, frames = demo["cylinder_xy"], np.arange(len(demo["cylinder_xy"]))
    before = np.nanmedian(xy[frames < phases["grasp"]], axis=0)
    after = np.nanmedian(xy[frames > phases["retract"]], axis=0)
    return TABLE_TO_ROBOT[:2, :2] @ (after - before)


def demo_plan(cfg: dict, demo, phases: dict[str, int], object_start: np.ndarray, goal: np.ndarray, rotate: bool = False):
    """Plan, duration and events for episode.run: the robot's own grasp, then the demonstration's carry.

    1. approach: the [pick_place] waypoints up to the grip (from above, thumb swung across, close until it holds).
       The robot's thumb can't copy a human side grasp, so the grasp itself isn't copied.
    2. carry: the demonstrator's hand path from lift-off (carry) to set-down (place), at the recorded speed, then
       still until the release (retract), with the
       hand's orientation and fingers held at the robot's grasp. The path starts at the simulated grasp and its end is
       blended onto the simulated goal, so the two tables may be laid out differently. Smoothed over [retarget]
       path_smoothing seconds; its height profile is kept but scaled to peak at [retarget] carry_height.
    3. release and rise: the [pick_place] waypoints after "lower", at the goal.
    rotate: turn the carry in the table plane to point at the goal (for goals in another direction than the clip's).
    """
    fps = 1 / (demo["time"][1] - demo["time"][0])
    waypoints = cfg["pick_place"]["waypoints"]
    grip = next(i for i, wp in enumerate(waypoints) if wp.get("until") == "grip")
    approach, t_grip, events = episode.waypoint_plan(cfg, object_start, goal, waypoints[:grip + 1])
    low = next(i for i, wp in enumerate(waypoints) if wp["name"] == "lower")
    after, t_after, _ = episode.waypoint_plan(cfg, object_start, goal, [{**waypoints[low], "duration": 0.0},
                                                                         *waypoints[low + 1:]])

    carry, place, retract = phases["carry"], phases["place"], phases["retract"]
    points, _ = hand_poses(demo["keypoints"], cfg["object"]["radius"], cfg["retarget"])
    points = _smooth(points @ TABLE_TO_ROBOT.T, max(1, round(cfg["retarget"]["path_smoothing"] * fps)))[carry:place + 1]
    grasp = object_start + [0, 0, cfg["grasp"]["height_above_center"]]
    blend = np.linspace(0, 1, len(points))[:, None]
    rel = points - points[0]
    if rotate:
        a = np.arctan2(*(goal - object_start)[1::-1]) - np.arctan2(*rel[-1, 1::-1])
        rel[:, :2] = rel[:, :2] @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    # The demo's path at the robot's scale: as long (in the table plane) as the simulated start-goal distance.
    rel[:, :2] *= np.linalg.norm((goal - object_start)[:2]) / max(np.linalg.norm(rel[-1, :2]), 1e-3)
    path = grasp + rel + blend * ((goal - object_start) - rel[-1])
    # ponytail: monocular height reads low (~5 cm for a ~10 cm lift), so keep the demo's height profile but scale its
    # peak to [retarget] carry_height; measure the lift from the cylinder instead if a clip needs its own height.
    rise = np.maximum(path[:, 2] - grasp[2], 0)
    path[:, 2] = grasp[2] + rise * cfg["retarget"]["carry_height"] / max(rise.max(), 1e-6)
    path[:, 2] += blend[:, 0] * waypoints[low]["up"]  # set down where [pick_place] lowers to, not pressed into the table
    # Set down, the hand stays put until the recorded release: a person nudges the object into place, which for the
    # robot means shoving it while gripped.
    path = np.vstack([path, np.repeat(path[-1:], retract - place, axis=0)])
    t_carry = (len(path) - 1) / fps

    rotation = scene.grasp_rotation(cfg)
    closed = np.array([cfg["grasp"]["fingers_closed"][n] for n in robot.RIGHT_HAND_DRIVES])

    def plan(t: float):
        if t <= t_grip:
            return approach(t)
        if t <= t_grip + t_carry:
            return path[round((t - t_grip) * fps)], rotation, closed
        return after(t - t_grip - t_carry)

    return plan, t_grip + t_carry + t_after, events


def _angle(R: np.ndarray) -> float:
    """Rotation angle [rad] of a rotation matrix."""
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about axis (any length; identity if ~0) by angle [rad] (Rodrigues)."""
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3)
    x, y, z = axis / n
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def standup_plan(cfg: dict, demo, phases: dict[str, int], fix: dict):
    """Stand-up (a lying cylinder picked palm-down and set upright on the goal) from one clip, following Human2Sim2Robot
    (arXiv 2504.12609) and VideoManip (arXiv 2602.09013). Returns (make_plan(start, goal), move, object_quat).

    From the clip: the cylinder's start and end (hand_tracking.find_cylinder) and its axis relative to the hand (across
    the thumb-finger pinch); the grasp point relative to the cylinder (the cylinder sits at the pinch centre, measured
    inside the hand, so the camera's depth error doesn't enter); the hand orientation; the slow descent while the
    fingers close; each finger's own closing curve (order and speed) and opening curve; the carry path and timing;
    the wrist turn that stands the cylinder up. The fingers stop where they meet the cylinder ([grasp]
    stop_at_contact: the support fingers), as a hand curls only until it meets the object.
    Not from the clip: the lift height ([retarget] carry_height) and the simulator's grasp correction fix (keys
    along, across, deeper [m], yaw, roll, pitch [deg], close, thumb_yaw, thumb_pitch; scripts/standup_grasp_search.py),
    which
    adjusts where the hand sits by a few cm and how far the fingers close (the robot hand isn't shaped like a
    human's; both papers need a step like it: contact optimisation, RL).
    """
    su = cfg["standup"]
    top, r_obj, h_obj = cfg["scene"]["table_height"], cfg["object"]["radius"], cfg["object"]["height"]
    fps = 1 / (demo["time"][1] - demo["time"][0])
    k = demo["keypoints"]
    pts, rots = hand_poses(k, r_obj, cfg["retarget"])
    place, retract = phases["place"], phases["retract"]
    # contact: the hand lowest on the cylinder before the place; lift-off: the fingertips su lift_off above that
    tips = np.nanmin(k[:, [4, 8, 12], 2], axis=1)
    contact = phases["grasp"] + int(np.nanargmin(tips[phases["grasp"]:place]))
    lift = contact + 1 + int(np.argmax(tips[contact + 1:place] > tips[contact] + su["lift_off"]))
    end = min(retract + round(su["release_seconds"] * fps), len(k) - 1)

    # The cylinder, tracked: lying before the hand reaches it, standing after the release.
    seen = np.flatnonzero(np.isfinite(demo["cylinder_yaw"][:max(lift - round(su["cylinder_seen_until"] * fps), 0)]))
    cyl = np.r_[np.nanmedian(demo["cylinder_xy"][seen], axis=0), r_obj]
    move = TABLE_TO_ROBOT[:2, :2] @ (np.nanmedian(demo["cylinder_xy"][retract:], axis=0) - cyl[:2])
    # Its axis relative to the hand: across the pinch at contact (the separately tracked yaw differs by 15-30 deg).
    pinch_line = np.nanmedian([k[i, 4] - k[i, [8, 12]].mean(0) for i in range(contact, contact + 10)], axis=0)
    axis = np.cross(TABLE_TO_ROBOT @ pinch_line, [0, 0, 1])
    axis /= np.linalg.norm(axis)
    obj_rot = np.column_stack([np.cross([0, 0, 1], axis), [0, 0, 1], axis])
    obj_rot[:, 0] /= np.linalg.norm(obj_rot[:, 0])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, obj_rot.ravel())

    # Hand depth: at contact the pinch touches the cylinder, so slide the hand along the camera ray through the pinch
    # until that point is nearest the cylinder; the same shift applies to the frames around it.
    mid = np.nanmedian([(k[i, 4] + k[i, [8, 12]].mean(0)) / 2 for i in range(contact, lift)], axis=0)
    eye = -demo["camera_from_table_R"].T @ demo["camera_from_table_t"].reshape(3)
    ray = (mid - eye) / np.linalg.norm(mid - eye)
    shift = eye + ray * ((cyl - eye) @ ray) - mid
    R = np.array([TABLE_TO_ROBOT @ r for r in rots])
    P = _smooth((pts + shift) @ TABLE_TO_ROBOT.T, max(1, round(cfg["retarget"]["path_smoothing"] * fps)))
    grasp_in_cyl = -(TABLE_TO_ROBOT @ (mid - pts[contact]))   # grasp point relative to the cylinder centre

    # Fingers: each follows its own curve, 0 = the clip's resting hand (first second; the tracker reads it partly
    # closed), 1 = at lift-off. A finger whose tracked range is too small to read (ring and pinky, hidden behind the
    # others) follows the mean of the readable ones. The closing starts where the thumb, index or middle finger first
    # passes su closing_onset of the way from its shape su closing_window s before contact to its shape at lift-off.
    names = list(demo["finger_joints"])  # the tracker's order (pinky first) -> the robot's drive order (thumb first)
    f = _smooth(demo["fingers"][:, [names.index(n) for n in robot.RIGHT_HAND_DRIVES]], max(1, round(0.2 * fps)))
    rest = np.nanmedian(f[:round(fps)], axis=0)
    closed = np.minimum(f[lift] * fix["close"], su["finger_max"])
    closed[:2] = fix["thumb_yaw"], fix["thumb_pitch"]
    span = f[lift] - rest
    readable = np.abs(span) > 0.05
    frac = np.clip((f - rest) / np.where(readable, span, 1.0), 0, 1)
    frac[:, ~readable] = frac[:, readable].mean(axis=1, keepdims=True)
    # Grasp rule (user, 2 Nov): thumb and index close together, on the clip's index curve; middle, ring and pinky
    # follow it [grasp] support_delay apart and aim for full curl, stopping where they touch ([grasp] stop_at_contact).
    step = round(cfg["grasp"]["support_delay"] * fps)
    frac[:, :2] = frac[:, [2]]
    for n, j in enumerate((3, 4, 5), start=1):
        frac[:, j] = np.r_[np.zeros(n * step), frac[:len(frac) - n * step, 2]]
    closed[3:] = np.asarray(su["finger_max"])[3:]
    w = max(contact - round(su["closing_window"] * fps), 0)
    reach = f[lift, :4] - f[w, :4]
    gain = (f[w:contact + 1, :4] - f[w, :4]) / np.where(np.abs(reach) > 0.05, reach, np.inf)
    close_start = w + int(np.argmax((gain > su["closing_onset"]).any(axis=1)))

    # correction of the hand's orientation: turned about the vertical (yaw), tilted about the hand's finger direction
    # (roll) and about its sideways axis (pitch); tilt is what one camera reads least reliably
    a, r_, p_ = np.radians([fix["yaw"], fix.get("roll", 0.0), fix.get("pitch", 0.0)])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(r_), -np.sin(r_)], [0, np.sin(r_), np.cos(r_)]])
    ry = np.array([[np.cos(p_), 0, np.sin(p_)], [0, 1, 0], [-np.sin(p_), 0, np.cos(p_)]])
    hold = rz @ R[contact] @ rx @ ry
    turn = [R[i] @ R[lift].T for i in range(len(R))]           # wrist turn since lift-off, applied to the grasp pose
    # The cylinder must end upright: the clip's turn can fall short (78 deg of the 90 needed, IMG_8510 start 8: it then
    # pivoted in the grip and landed 7.9 cm off), so the smallest rotation that makes the held cylinder's axis vertical
    # at the set-down is added in step with the clip's own turn (0 at lift-off, all of it where the turn ends).
    end_axis = turn[place] @ axis
    up = np.array([0.0, 0.0, 1.0 if end_axis[2] >= 0 else -1.0])
    k = np.cross(end_axis, up)
    missing = np.arctan2(np.linalg.norm(k), end_axis @ up)
    total = _angle(turn[place])
    for i in range(lift, len(turn)):
        turn[i] = _rotation(k, missing * min(_angle(turn[i]) / max(total, 1e-6), 1.0) if i <= place else missing) @ turn[i]
    nudge = fix["along"] * axis + fix["across"] * np.cross([0, 0, 1], axis) - [0, 0, fix["deeper"]]
    # The hand comes down first, fingers open, then the video's closing plays out at the grasp height (the robot's
    # longer fingers, closing on the way down as the demonstrator's do, landed on top of the cylinder).
    drop = np.zeros(lift + 1 - close_start)
    t_down, t_close = su["descent_seconds"], (lift - close_start) / fps
    t_carry, t_rel = (place - lift) / fps, (end - place + 1) / fps

    def make_plan(object_start: np.ndarray, goal: np.ndarray):
        c0 = np.r_[object_start[:2], top + r_obj]
        # place_x/y: where the stood-up cylinder lands relative to where the plan aims, measured in the simulator (it
        # pivots in the thumb-index pinch while turning); the hand aims that much the other way
        c1 = np.r_[goal[:2] - [fix.get("place_x", 0.0), fix.get("place_y", 0.0)], top + h_obj / 2 + 0.005]
        site = c0 + grasp_in_cyl + nudge
        above = site + [0, 0, su["approach_height"]]
        # carry: the video's path from lift-off to set-down, scaled onto this start -> goal, its wrist turn on top
        rel = P[lift:place + 1] - P[lift]
        goal_site = c1 + turn[place] @ (site - c0)
        blend = np.linspace(0, 1, len(rel))[:, None]
        path = site + rel + blend * ((goal_site - site) - rel[-1])
        rise = np.maximum(path[:, 2] - np.minimum(site[2], goal_site[2]), 0)
        path[:, 2] += rise * cfg["retarget"]["carry_height"] / max(rise.max(), 1e-6) - rise
        release = path[-1] + (P[place:end + 1] - P[place])
        release[:, 2] = np.maximum(release[:, 2], path[-1][2])
        start_close = site + [0, 0, drop[0]]

        def plan(t: float):
            if t < t_down:                                     # down to where the closing starts, hand open
                return above + t / t_down * (start_close - above), hold, np.zeros(6)
            t -= t_down
            if t < t_close:                                    # the video's slow, finger-by-finger closing
                i = min(round(t * fps), len(drop) - 1)
                return site + [0, 0, drop[i]], hold, frac[close_start + i] * closed
            t -= t_close
            if t < t_carry:
                i = round(t * fps)
                return path[i], turn[lift + i] @ hold, closed
            t -= t_carry                                       # release: each finger opens as in the video
            i = min(round(t * fps), len(release) - 1)
            return release[i], turn[place + i] @ hold, frac[place + i] * closed

        return plan, t_down + t_close + t_carry + t_rel, {"grip": t_down + t_close}

    return make_plan, move, quat


if __name__ == "__main__":
    # Self-check: a synthetic demo (open hand, close, lift 10 cm, put down, open) is segmented at the right frames.
    cfg = {"smoothing_frames": 5, "closed_fraction": 0.5, "lift_height": 0.02}
    t = np.arange(300)
    closed = ((t >= 60) & (t < 240) | (t < 30)).astype(float)  # t < 30: resting hand misread as closed
    height = 0.03 + 0.10 * np.clip(np.minimum(t - 100, 200 - t) / 20, 0, 1)
    phases = label_phases(height, closed, cfg)
    expected = {"reach": 0, "grasp": 60, "carry": 105, "place": 196, "retract": 240}
    assert all(abs(phases[k] - v) <= 3 for k, v in expected.items()), phases

    # And the demo's carry starts at the simulated grasp and ends at the simulated goal.
    hand = np.zeros((21, 3))
    hand[[5, 9, 13, 17]] = [[0.08, 0.02, 0.0], [0.09, 0.0, 0.0], [0.085, -0.02, 0.0], [0.075, -0.04, 0.0]]
    path = np.stack([np.linspace(0, 0.3, 300), np.zeros(300), height], axis=1)
    demo = {"keypoints": hand[None] + path[:, None], "time": t / 30.0, "finger_joints": robot.RIGHT_HAND_DRIVES,
            "fingers": np.zeros((300, 6))}
    from g1_imitation import config

    cfg = config.load()
    lift = np.array([0, 0, cfg["grasp"]["height_above_center"]])
    object_start, goal = np.array([0.27, -0.18, 0.775]), np.array([0.27, -0.06, 0.775])
    plan, duration, events = demo_plan(cfg, demo, phases, object_start, goal)
    t_end = events["grip"] + (phases["retract"] - phases["carry"]) / 30.0
    assert np.allclose(plan(events["grip"])[0], object_start + lift, atol=1e-6), plan(events["grip"])[0]
    set_down = goal + lift + [0, 0, cfg["pick_place"]["waypoints"][-3]["up"]]
    assert np.allclose(plan(t_end)[0], set_down, atol=1e-6), plan(t_end)[0]
    assert np.allclose(plan(duration)[0][:2], goal[:2], atol=1e-6)  # rises straight up from the goal
    peak = max(plan(t)[0][2] for t in np.arange(events["grip"], t_end, 0.01)) - object_start[2] - lift[2]
    assert abs(peak - cfg["retarget"]["carry_height"]) < 0.005, peak
    print("ok", phases)
