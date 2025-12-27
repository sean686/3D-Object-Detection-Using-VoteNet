import os
import json
import numpy as np
from plyfile import PlyData

# ================= CONFIG =================
JSON_DIR = "data/json"
PLY_DIR  = "data/ply"
OUT_DIR  = "data/votenet_format"

CLASS_NAME = "wire"
CLASS_ID = 0
NUM_POINTS = 20000

os.makedirs(OUT_DIR, exist_ok=True)

# ================= UTILS =================
def load_ply_xyzrgb(path):
    ply = PlyData.read(path)
    v = ply['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
    if {'red','green','blue'}.issubset(v.data.dtype.names):
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1) / 255.0
    else:
        rgb = np.zeros_like(xyz)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)

def load_labelcloud_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    boxes = []
    for obj in data.get("objects", []):
        if obj.get("name") != CLASS_NAME:
            continue

        cx = obj["centroid"]["x"]
        cy = obj["centroid"]["y"]
        cz = obj["centroid"]["z"]

        l = obj["dimensions"]["length"]
        w = obj["dimensions"]["width"]
        h = obj["dimensions"]["height"]

        boxes.append([cx, cy, cz, l, w, h, 0.0, CLASS_ID])

    return np.array(boxes, dtype=np.float32)

def points_in_box(points, center, size):
    cx, cy, cz = center
    l, w, h = size
    x, y, z = points[:,0], points[:,1], points[:,2]

    return (
        (x >= cx - l/2) & (x <= cx + l/2) &
        (y >= cy - w/2) & (y <= cy + w/2) &
        (z >= cz - h/2) & (z <= cz + h/2)
    )

# ================= MAIN =================
json_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith(".json"))

for idx, jf in enumerate(json_files):
    scene_id = f"{idx:06d}"
    print(f"Processing {scene_id}")

    pc = load_ply_xyzrgb(os.path.join(PLY_DIR, jf.replace(".json",".ply")))
    boxes = load_labelcloud_json(os.path.join(JSON_DIR, jf))

    if pc.shape[0] > NUM_POINTS:
        choice = np.random.choice(pc.shape[0], NUM_POINTS, replace=False)
        pc = pc[choice]

    N = pc.shape[0]
    point_votes = np.zeros((N, 3), dtype=np.float32)
    vote_mask   = np.zeros((N,), dtype=np.int64)

    for b in boxes:
        center = b[0:3]
        size = b[3:6]
        mask = points_in_box(pc[:,0:3], center, size)

        offsets = center - pc[mask, 0:3]
        point_votes[mask] = offsets
        vote_mask[mask] = 1

    np.savez_compressed(f"{OUT_DIR}/{scene_id}_pc.npz", pc=pc)
    np.save(f"{OUT_DIR}/{scene_id}_bbox.npy", boxes)
    np.savez_compressed(
        f"{OUT_DIR}/{scene_id}_votes.npz",
        point_votes=point_votes,
        vote_mask=vote_mask
    )

    print(f"  boxes={len(boxes)} points={pc.shape[0]}")

print("✔ VoteNet data generation complete")

