"""Phone video -> 3D hand keypoints in the table frame.

WiLoR (scripts/track_hands_wilor.py) gives 21 hand keypoints per frame, both in pixels and as a metric 3D hand shape
centered on the hand. Fitting that 3D shape to the pixel keypoints (solvePnP) places the hand in front of the camera, and the ArUco
markers taped to the table ([hand_tracking] markers) give the camera's pose relative to the table.

Sheet frame: origin and axes as laid out in [hand_tracking] markers (for data/aruco_board_a4.pdf: the sheet's center,
x toward its right edge, y toward its top edge), z up. Table frame: the same, turned so that x points to the
demonstrator's right and y away from them (sheet_to_table).
"""

import cv2
import numpy as np


def camera_matrix(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    """Pinhole intrinsics from the field of view (no lens distortion)."""
    f = width / 2 / np.tan(np.radians(horizontal_fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])


def marker_points(frame: np.ndarray, cfg: dict):
    """Corners of every visible marker of [hand_tracking] markers: (sheet-frame points N x 3, pixels N x 2), or None."""
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["marker_dictionary"])))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    s = cfg["marker_size"] / 2
    square = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])  # ArUco corner order, z out of the marker
    found = [(square + [*cfg["markers"][str(i)], 0], c[0]) for i, c in zip([] if ids is None else ids.ravel(), corners)
             if str(i) in cfg["markers"]]
    return tuple(np.concatenate(x) for x in zip(*found)) if found else None


def fit_camera(views: list, width: int, height: int) -> np.ndarray:
    """Camera matrix fitted to marker views [(points, pixels), ...] of a still camera: one focal length, principal
    point at the image center, no distortion. The phone's field of view changes with its video mode and
    stabilization (56.8-65.0 deg in our clips), so it is fitted per clip; needs the sheet seen at an angle."""
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    points, pixels = ([np.float32(v[i]) for v in views] for i in (0, 1))
    return cv2.calibrateCamera(points, pixels, (width, height), camera_matrix(width, height, 65), None, flags=flags)[1]


def marker_pose(frame: np.ndarray, K: np.ndarray, cfg: dict):
    """Sheet pose in the camera frame, (R, t) with x_cam = R @ x_sheet + t, from every visible marker (more markers,
    steadier pose; one is enough), or None if none is visible."""
    found = marker_points(frame, cfg)
    if found is None:
        return None
    _, rvec, tvec = cv2.solvePnP(*found, K, None, flags=cv2.SOLVEPNP_IPPE)
    return cv2.Rodrigues(rvec)[0], tvec.ravel()


def sheet_to_table(resting_keypoints: np.ndarray) -> np.ndarray:
    """Rotation (about z, a multiple of 90 deg) from the sheet frame to the table frame (x to the demonstrator's right,
    y away from them), from a hand resting flat at the start of the clip: its fingers point away from the
    demonstrator. So it doesn't matter which way the sheet was taped down."""
    fingers = np.nanmean(resting_keypoints[:, 9, :2] - resting_keypoints[:, 0, :2], axis=0)  # wrist -> middle base
    a = np.radians(90 * round((90 - np.degrees(np.arctan2(fingers[1], fingers[0]))) / 90))
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def cylinder_outline(xy: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, radius: float, height: float,
                     yaw: float | None = None):
    """Pixel outline (convex hull) of a cylinder at table point xy: upright, or lying with its axis at yaw [rad]."""
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    if yaw is None:
        rim = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)], axis=1) + [*xy, 0]
        points = np.vstack([rim, rim + [0, 0, height]])
    else:
        axis, side = np.array([np.cos(yaw), np.sin(yaw), 0]), np.array([-np.sin(yaw), np.cos(yaw), 0])
        rim = radius * (np.cos(a)[:, None] * side + np.sin(a)[:, None] * [0, 0, 1]) + [*xy, radius]
        points = np.vstack([rim - axis * height / 2, rim + axis * height / 2])
    return cv2.convexHull(cv2.projectPoints(points, cv2.Rodrigues(R)[0], t, K, None)[0].astype(np.int32))


def find_cylinder(frame, R, t, K, radius: float, height: float, cfg: dict):
    """(table point [x, y] under the cylinder's centre, yaw) of a dark cylinder on the table, or None if none is fully
    visible; yaw is None when it stands upright, else the lying cylinder's axis direction [rad, 0..pi). Each candidate
    pose (upright, or lying at every cfg lying_step_deg) is fitted as below and the best overlap wins.

    Dark blobs (HSV value below cfg max_value) are candidates; each is placed on the table where the projected
    cylinder's outline has the blob's centroid (found by casting centroids onto the plane at half the cylinder's height
    and correcting a few times), and kept if its overlap (IoU) with that outline is at least cfg min_iou. That rejects the black marker squares, dark table edges, and a cylinder partly covered by
    the hand; blobs farther than cfg max_distance from the table origin (the background) are ignored. ponytail: a lifted cylinder lands up to a lift height off (it is cast at standing height); during the
    carry use the hand instead.
    """
    dark = (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2] < cfg["max_value"]).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    best, best_iou = None, cfg["min_iou"]
    origin = -R.T @ t  # camera center, table frame

    def cast(pixel, z):  # pixel -> table point at height z
        ray = R.T @ np.linalg.solve(K, [*pixel, 1])
        return (origin + ray * (z - origin[2]) / ray[2])[:2]

    poses = [None, *np.radians(np.arange(0, 180, cfg["lying_step_deg"]))]
    for i in np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] > cfg["min_pixels"]) + 1:
        bx, by, bw, bh = stats[i, :4]
        for yaw in poses:
            z = height / 2 if yaw is None else radius  # the centre's height
            target = cast(centroids[i], z)
            xy = target
            for _ in range(4):
                m = cv2.moments(cylinder_outline(xy, R, t, K, radius, height, yaw))
                xy = xy + target - cast([m["m10"] / m["m00"], m["m01"] / m["m00"]], z)
            poly = cylinder_outline(xy, R, t, K, radius, height, yaw)
            # the overlap only counts pixels inside the blob or the outline: compare within their joint bounding box
            # (the same IoU as on the whole frame, without building full-size masks per candidate)
            lo, hi = poly.reshape(-1, 2).min(0), poly.reshape(-1, 2).max(0) + 1  # convexHull points are N x 1 x 2
            x0, y0 = max(0, min(bx, lo[0])), max(0, min(by, lo[1]))
            x1, y1 = min(dark.shape[1], max(bx + bw, hi[0])), min(dark.shape[0], max(by + bh, hi[1]))
            blob = labels[y0:y1, x0:x1] == i
            outline = np.zeros(blob.shape, np.uint8)
            cv2.fillConvexPoly(outline, poly - [x0, y0], 1)
            iou = (blob & (outline > 0)).sum() / (blob | (outline > 0)).sum()
            if iou > best_iou and np.linalg.norm(xy) < cfg["max_distance"]:
                best, best_iou = (xy, yaw), iou
    return best


def hand_in_camera(pixels: np.ndarray, shape: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Places the metric hand shape so it projects onto the pixel keypoints. Returns 21x3 points, camera frame."""
    _, rvec, tvec = cv2.solvePnP(shape, pixels, K, None, flags=cv2.SOLVEPNP_SQPNP)
    return shape @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()


if __name__ == "__main__":
    # Self-check on synthetic data: a marker and a hand shape seen by a known camera are recovered exactly.
    from g1_imitation import config

    cfg = config.load()["hand_tracking"]
    K = camera_matrix(1920, 1080, 69)
    R_true = cv2.Rodrigues(np.array([2.3, 0.1, 0.05]))[0]  # camera looking down at the table at ~45 deg
    t_true = np.array([0.02, -0.03, 0.6])

    # The marker sheet as an image (4000 px per metre, 0.3 m square, centered on the table origin), seen by the camera,
    # with one marker hidden as if under the hand.
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["marker_dictionary"]))
    px, half = 4000, 0.15
    sheet = np.full((round(2 * half * px),) * 2, 255, np.uint8)
    size = round(cfg["marker_size"] * px)
    for i, (x, y) in list(cfg["markers"].items())[:-1]:
        col, row = round((x + half) * px - size / 2), round((half - y) * px - size / 2)
        sheet[row:row + size, col:col + size] = cv2.aruco.generateImageMarker(dictionary, int(i), size)
    page = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    projected = cv2.projectPoints(page, cv2.Rodrigues(R_true)[0], t_true, K, None)[0].reshape(4, 2)
    n = sheet.shape[0]
    warp = cv2.getPerspectiveTransform(np.float32([[0, 0], [n, 0], [n, n], [0, n]]), np.float32(projected))
    frame = cv2.cvtColor(cv2.warpPerspective(sheet, warp, (1920, 1080), borderValue=200), cv2.COLOR_GRAY2BGR)
    R, t = marker_pose(frame, K, cfg)
    fitted = fit_camera([marker_points(frame, cfg)], 1920, 1080)
    assert abs(fitted[0, 0] / K[0, 0] - 1) < 0.02, (fitted[0, 0], K[0, 0])
    # A resting hand whose fingers point along the sheet's +x: the table frame is the sheet turned 90 deg.
    hand_rest = np.zeros((5, 21, 3))
    hand_rest[:, 9, 0] = 0.09
    assert np.allclose(sheet_to_table(hand_rest) @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    angle = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    assert np.linalg.norm(t - t_true) < 0.005 and angle < 1.0, (t - t_true, angle)

    hand = np.random.default_rng(0).uniform(-0.08, 0.08, (21, 3))
    pixels = cv2.projectPoints(hand, cv2.Rodrigues(R_true)[0], t_true, K, None)[0].reshape(21, 2)
    assert np.allclose(hand_in_camera(pixels, hand, K), hand @ R_true.T + t_true, atol=1e-4)
    # A dark cylinder on a bright table is found where it stands; a dark strip of the same area is not taken for it.
    image = np.full((1080, 1920, 3), 200, np.uint8)
    cv2.fillConvexPoly(image, cylinder_outline(np.array([0.1, -0.05]), R_true, t_true, K, 0.025, 0.15), (20, 20, 20))
    cv2.rectangle(image, (100, 900), (700, 930), (20, 20, 20), -1)
    xy = find_cylinder(image, R_true, t_true, K, 0.025, 0.15, cfg["cylinder"])
    assert xy is not None and np.linalg.norm(xy - [0.1, -0.05]) < 0.01, xy
    image[:] = 200
    cv2.rectangle(image, (100, 900), (700, 930), (20, 20, 20), -1)
    assert find_cylinder(image, R_true, t_true, K, 0.025, 0.15, cfg["cylinder"]) is None

    print(f"ok, marker position error {1000 * np.linalg.norm(t - t_true):.1f} mm, rotation error {angle:.2f} deg")
