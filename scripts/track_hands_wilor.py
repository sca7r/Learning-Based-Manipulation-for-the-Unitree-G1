"""Right-hand keypoints with WiLoR (via WiLoR-mini), for a GPU machine. Standalone: needs only wilor-mini, torch, OpenCV.

  python scripts/track_hands_wilor.py data/raw/IMG_*.MOV

Writes data/hands/<clip>.npz per clip: pixels (frames x 21 x 2, full-image pixels) and shape (frames x 21 x 3, metres,
the hand's 3D shape in its own frame), NaN where no right hand was found. Copy data/hands/ back to the main machine;
scripts/process_demos.py reads them.
Keypoint order is OpenPose's: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
WiLoR fits a full 3D hand model, so it also finds hands that wrap around the object.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "hands")
    args = parser.parse_args()
    gpu = torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0) if gpu else "none, running on CPU (slow)")
    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda" if gpu else "cpu"),
                                             dtype=torch.float16 if gpu else torch.float32, verbose=False)  # half precision only on the GPU
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        video = cv2.VideoCapture(str(clip))
        pixels, shape = [], []
        while True:
            ok, frame = video.read()
            if not ok:
                break
            # WiLoR-mini blurs the whole frame on the CPU before cropping the hand; at 4K that, not the GPU, is the
            # bottleneck. 1280 px wide still leaves the hand ~200 px tall; pixels are scaled back below.
            scale = min(1.0, 1280 / frame.shape[1])
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right = [h for h in pipe.predict(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)) if h["is_right"] == 1]
            if right:  # ponytail: the biggest right hand; the demonstrator's is the only one in frame
                x0, y0, x1, y1 = zip(*[h["hand_bbox"] for h in right])
                h = right[int(np.argmax((np.array(x1) - x0) * (np.array(y1) - y0)))]["wilor_preds"]
                pixels.append(h["pred_keypoints_2d"][0] / scale)
                shape.append(h["pred_keypoints_3d"][0])
            else:
                pixels.append(np.full((21, 2), np.nan))
                shape.append(np.full((21, 3), np.nan))
            if len(pixels) % 100 == 0:
                print(f"{clip.name}: frame {len(pixels)}", flush=True)
        np.savez(args.out / f"{clip.stem}.npz", pixels=np.array(pixels), shape=np.array(shape))
        found = (~np.isnan(np.array(pixels)[:, 0, 0])).sum()
        print(f"{clip.name}: right hand in {found}/{len(pixels)} frames -> {args.out / clip.stem}.npz")


if __name__ == "__main__":
    main()
