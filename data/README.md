# Collection note

These are phone videos of my right hand moving a cylinder on a table. Every robot demonstration in this project
comes from them: the clip sets where the cylinder goes, how it is carried, and how the fingers grip and when they
close.

## What is in this folder

```
data/
  aruco_board_a4.pdf      the marker sheet: 4 ArUco markers on A4, gives the camera pose and the metric scale
  pickandplace.csv        one row per clip: session, task, success, the cylinder's measured move, notes
  phases.csv              first frame of each phase per clip (reach, grasp, carry, place, retract)
  standup_grasp.csv       per stand-up clip: the grasp correction found in simulation (scripts/standup_grasp_search.py)
  raw/                    sample clips IMG_8465, IMG_8473, IMG_8484 (the full set is 1.6 GB, not in git)
  hands/                  hand keypoints of the sample clips (scripts/track_hands_wilor.py)
  processed/              all 47 usable clips after scripts/process_demos.py: hand keypoints and orientation on the
                          table, finger angles, cylinder position, camera pose
  sample_episodes/        3 robot replays as they are saved for training: head and wrist camera video, joint
                          states and targets (scripts/replay_demos.py --record)
```

Generated data is not in git: all replays go to `data/episodes/`, the LeRobot datasets to `data/lerobot/`.

## The recordings

| | |
|---|---|
| Phone | iPhone 15 Pro Max on a tripod, about 45 degrees above the table |
| Sessions | 3; the camera was moved between sessions (field of view fitted per clip: 56.8, 62.9 and 65.0 degrees) |
| Video | 45 clips at 1920 x 1080, 60 fps; 4 clips at 3840 x 2160, 30 fps |
| Object | black cylinder, 50 mm across, 150 mm tall; mass not measured (the simulation uses 250 g) |


Pick-and-place: the cylinder stands at the start, I pick it up, lift it about 10 cm and put it down upright on the
goal. Stand-up: the cylinder lies on its side, I pick it up with thumb and fingers (the palm stays off it), turn the
wrist and stand it upright on the goal.

## Recording more clips

**Scene**
- A plain, matte table. No glass and no busy patterns.
- Print `aruco_board_a4.pdf` at 100 % ("actual size"); each black square and the printed line measure 60 mm. Tape
  it flat, between the start and the goal, and don't move it during a session. Any one marker fully in view is
  enough, so a hand covering the others is fine.
- A cylinder about 150 mm tall and 50 mm across, in a colour that stands out from the table and your skin. Weigh it.

**Camera**
- Tripod, about 70 to 80 cm from the cylinder, looking down at about 45 degrees. Not straight down: from above, the
  hand's height can't be measured.
- The marker sheet, the cylinder, the goal and your whole right hand and forearm in every frame.
- Main lens, no zoom. 1080p at 30 or 60 fps. Turn off video stabilisation, HDR and night mode: stabilisation warps
  the image and breaks the camera-to-table geometry.
- Lock focus and exposure before the first clip. Bright, even light, not from behind the table.

**Hand**
- Right hand only; keep the left hand off the table. Roll up the sleeve, take off a watch or rings.
- Move calmly, a bit slower than normal.

**One clip**
1. Start recording with your hand resting flat on the table, still for 1 second.
2. Do the task once.
3. Rest the hand flat again for 1 second, then stop.
4. Add a row to `pickandplace.csv`. Keep failed takes too, with `success` set to 0.

Copy the original files over USB or a cloud drive at original quality. Messaging apps re-compress the video and drop
its metadata.

## Phase labels

`python scripts/process_demos.py data/raw/IMG_8473.MOV` labels a clip's phases from how closed the fingers are and
how high the wrist is, and writes `outputs/demos/<clip>.mp4` with the frame number and phase drawn on it. Watch it;
if a phase starts on the wrong frame, fix the number in `phases.csv` and run it again. Rows already in the file are
never overwritten.
