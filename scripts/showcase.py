"""Good-looking videos of recorded trials for the README: replays a trial's saved poses (qpos) in a dressed-up copy of
the scene (sky, tiled floor, a table with a top and legs instead of the solid box, a key light with shadows) and films
it from six cameras at once (two wide, front, side, top, and a close one that follows the cylinder). Looks only:
nothing is simulated, and the policies' scene and cameras (scene.build_model) are untouched, so the trained policies see exactly what they were trained on.

  MUJOCO_GL=glfw python scripts/showcase.py outputs/evaluate/IMG_8465_0.npz --out outputs/showcase --gif
Takes the evaluate.py trial files (<clip>_<trial>.npz: qpos per frame, start, goal, fps).
"""

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from g1_imitation import config, scene

SIZE = (640, 360)  # each view [width, height]; the video is a 3 x 2 grid of the views
# Free cameras: lookat [x, y, z] (None: follow the cylinder), distance m, azimuth and elevation deg.
CAMERAS = {"wide left": ([0.28, -0.12, 0.82], 1.55, 150, -22), "wide right": ([0.28, -0.12, 0.82], 1.55, 210, -22),
           "front": ([0.30, -0.10, 0.85], 1.8, 180, -15), "side": ([0.28, -0.12, 0.82], 1.4, 250, -20),
           "top": ([0.32, -0.15, 0.80], 1.2, 180, -65), "close": (None, 0.55, 205, -28)}


def dress(spec: mujoco.MjSpec, cfg: dict) -> None:
    """Sky, floor tiles, table top and legs, materials and lights. Visual-only geoms (no contacts)."""
    sc = cfg["scene"]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.74, 0.88], rgb2=[0.12, 0.16, 0.24], width=512, height=512)
    spec.add_texture(name="tiles", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.80, 0.78, 0.74], rgb2=[0.70, 0.68, 0.64], width=512, height=512)
    floor = spec.add_material(name="floor", texrepeat=[6, 6], reflectance=0.08)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tiles"
    spec.add_material(name="wood", rgba=[0.60, 0.43, 0.28, 1], specular=0.3, shininess=0.4)
    spec.add_material(name="metal", rgba=[0.25, 0.25, 0.27, 1], specular=0.6, shininess=0.8)
    spec.add_material(name="paint", rgba=[0.85, 0.15, 0.10, 1], specular=0.5, shininess=0.6)
    spec.geom("floor").material, spec.geom("floor").rgba = "floor", [1, 1, 1, 1]
    spec.geom("object").material = "paint"

    table = spec.geom("table")
    table.rgba = [0, 0, 0, 0]  # the physical box stays (same model), hidden
    (cx, cy), (hx, hy), top, slab = sc["table_center"], sc["table_half_size"], sc["table_height"], 0.03
    world = spec.worldbody
    world.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, top - slab / 2],
                   size=[hx, hy, slab / 2], material="wood", contype=0, conaffinity=0)
    for sx in (-1, 1):  # four legs, 5 cm in from the corners
        for sy in (-1, 1):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[cx + sx * (hx - 0.05), cy + sy * (hy - 0.05),
                           (top - slab) / 2], size=[0.02, (top - slab) / 2, 0], material="metal", contype=0, conaffinity=0)
    # one spot light from the front left casts the shadows; the headlight is turned down so they show
    world.add_light(pos=[1.2, 0.8, 2.2], dir=[-0.5, -0.35, -0.8], diffuse=[0.6, 0.6, 0.6], castshadow=True,
                    type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=60)
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.25, 0.25, 0.25], [0.35, 0.35, 0.35]
    spec.visual.quality.shadowsize = 4096
    spec.visual.global_.offwidth, spec.visual.global_.offheight = SIZE


def camera(name: str, data: mujoco.MjData, model: mujoco.MjModel) -> mujoco.MjvCamera:
    """One of CAMERAS for the current frame; the close camera moves with the cylinder."""
    lookat, distance, azimuth, elevation = CAMERAS[name]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = scene.object_pos(model, data) if lookat is None else lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def render(cfg: dict, trial: Path, out: Path, gif: bool) -> Path:
    """Write <trial>.mp4 (and <trial>.gif) into out and return the video's path."""
    rec = np.load(trial)
    spec = scene.build_spec(cfg, rec["start"], rec["goal"])  # the cylinder's pose comes from qpos (stand-up too)
    dress(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, SIZE[1], SIZE[0])
    path = out / f"{trial.stem}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(rec["fps"]), (3 * SIZE[0], 2 * SIZE[1]))
    small = []
    for i, qpos in enumerate(rec["qpos"]):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        views = []
        for name in CAMERAS:
            renderer.update_scene(data, camera=camera(name, data, model))
            views.append(renderer.render())
        frame = np.vstack([np.hstack(views[:3]), np.hstack(views[3:])])  # 3 views on top, 3 below
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if gif and i % 3 == 0:  # ~8 frames/s at 25 Hz, a third of the size: small enough for a README
            small.append(Image.fromarray(frame).resize((frame.shape[1] // 3, frame.shape[0] // 3)))
    writer.release()
    renderer.close()
    if small:
        small[0].save(out / f"{trial.stem}.gif", save_all=True, append_images=small[1:], duration=120, loop=0)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", type=Path, nargs="+", help="evaluate.py trial files (<clip>_<trial>.npz)")
    parser.add_argument("--out", type=Path, default=Path("outputs/showcase"))
    parser.add_argument("--gif", action="store_true", help="also a small GIF for the README")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    for t in args.trials:
        print(render(cfg, t, args.out, args.gif))
