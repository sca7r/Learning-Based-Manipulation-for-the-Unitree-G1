#!/usr/bin/env bash
# Downloads the pinned external models into assets/ (gitignored): robot model, balance policy, hand URDF.
set -euo pipefail
cd "$(dirname "$0")/.."

# G1 29-DoF + Inspire hands MJCF, from supermimic (MIT, ETH Zurich).
SUPERMIMIC_SHA=f8f7aebaad9dea3a3ccfa8ef0676e1ff157e3442
mkdir -p assets/g1_inspire
curl -fsSL "https://codeload.github.com/xingzewu24/supermimic_vibed/tar.gz/${SUPERMIMIC_SHA}" \
  | tar -xz -C assets/g1_inspire --strip-components=5 \
      "supermimic_vibed-${SUPERMIMIC_SHA}/data/tracking/robot/g1_inspire"

# Pretrained G1 + Inspire locomotion policy, from josabb (Apache-2.0), used for standing balance.
JOSABB_REV=83e06e1277b30f9a694678b711c8c753c03107a6
HF="https://huggingface.co/josabb/G1-humanoid-6dof-hands-locomotion-rl/resolve/${JOSABB_REV}"
mkdir -p assets/balance_policy
for f in onnx/policy.onnx configs/env.yaml; do  # env.yaml: the training config robot.py copies gains from
  curl -fsSL -o "assets/balance_policy/$(basename "$f")" "${HF}/${f}"
done

# Inspire hand URDF for finger retargeting, from dex-urdf (dexsuite).
DEX_URDF_SHA=f5e7132f22108164577fea4c25ef99b5cc0e1900
DEX="https://raw.githubusercontent.com/dexsuite/dex-urdf/${DEX_URDF_SHA}/robots/hands/inspire_hand"
mkdir -p assets/dex_urdf/inspire_hand/meshes/{visual,collision}
curl -fsSL -o assets/dex_urdf/inspire_hand/inspire_hand_right.urdf "${DEX}/inspire_hand_right.urdf"
for mesh in $(grep -o 'meshes/[a-z]*/[a-z_]*\.[a-z]*' assets/dex_urdf/inspire_hand/inspire_hand_right.urdf | sort -u); do
  curl -fsSL -o "assets/dex_urdf/inspire_hand/${mesh}" "${DEX}/${mesh}"
done

echo "assets/ ready:"; ls assets/*
