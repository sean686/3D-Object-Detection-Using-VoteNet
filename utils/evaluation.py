import os
import numpy as np
import torch
import torch.nn as nn
import pyvista as pv

from wire_config import WireDatasetConfig

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data/votenet_format")
CHECKPOINT = os.path.join(BASE_DIR, "trained_model", "votenet_epoch_10.pth")

SCORE_THRESH = 0.03        # LOW for thin wires
NMS_IOU_THRESH = 0.10      # LOW to avoid suppressing nearby wires
NUM_PROPOSAL = 256

os.makedirs("eval_results", exist_ok=True)

cfg = WireDatasetConfig()
MEAN_SIZE = cfg.mean_size_arr[0]  # (3,)

# =========================
# DATASET
# =========================
class WireVoteNetDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ids = sorted([
            f.replace("_pc.npz", "")
            for f in os.listdir(data_dir)
            if f.endswith("_pc.npz")
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        pc = np.load(os.path.join(self.data_dir, f"{sid}_pc.npz"))["pc"][:, :3]  # FIX: only xyz
        boxes = np.load(os.path.join(self.data_dir, f"{sid}_bbox.npy"))
        return pc, boxes, sid

# =========================
# MODEL (MATCH train.py)
# =========================
class VoteNet(nn.Module):
    def __init__(self, num_proposal=NUM_PROPOSAL):
        super().__init__()
        self.num_proposal = num_proposal

        # ---------- Simple Backbone (matches your training.py structure) ----------
        from pointnet2.pointnet2_modules import PointnetSAModuleVotes

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.02,
            nsample=16,
            mlp=[0, 64, 64, 128],
            use_xyz=True
        )
        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.04,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True
        )
        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.08,
            nsample=32,
            mlp=[256, 256, 256, 256],
            use_xyz=True
        )

        # ---------- Vote Layer ----------
        self.vote_layer = nn.Conv1d(256, 256 + 3, 1)

        # ---------- Proposal Head ----------
        self.head = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
        )

        self.obj = nn.Conv1d(128, 1, 1)
        self.center = nn.Conv1d(128, 3, 1)
        self.size_residual = nn.Conv1d(128, 3, 1)

    def forward(self, pc):
        """
        pc: (B, N, 3)
        """
        # ENSURE INPUT HAS 3 CHANNELS
        if pc.shape[2] > 3:
            pc = pc[:, :, :3]

        B, N, _ = pc.shape
        xyz = pc
        features = None

        # ---------- Backbone ----------
        xyz, features, _ = self.sa1(xyz, features)
        xyz, features, _ = self.sa2(xyz, features)
        xyz, features, _ = self.sa3(xyz, features)

        # ---------- Voting ----------
        vote = self.vote_layer(features)
        vote_feat = vote[:, :256, :]
        vote_offset = vote[:, 256:, :]
        vote_xyz = xyz + vote_offset.transpose(1, 2)

        # ---------- Sample Proposals ----------
        idx = torch.randperm(vote_xyz.shape[1], device=pc.device)[:self.num_proposal]
        idx = idx.unsqueeze(0).repeat(B, 1)

        proposal_xyz = torch.stack([vote_xyz[b, idx[b]] for b in range(B)])
        proposal_feat = torch.stack([vote_feat[b, :, idx[b]] for b in range(B)])

        # ---------- Prediction ----------
        x = self.head(proposal_feat)

        return {
            "objectness": self.obj(x).squeeze(1),
            "center": self.center(x).transpose(1, 2),
            "size_residual": self.size_residual(x).transpose(1, 2)
        }

# =========================
# IoU & NMS
# =========================
def iou_3d(c1, s1, c2, s2):
    min1, max1 = c1 - s1 / 2, c1 + s1 / 2
    min2, max2 = c2 - s2 / 2, c2 + s2 / 2
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter)
    vol1 = np.prod(s1)
    vol2 = np.prod(s2)
    return inter_vol / (vol1 + vol2 - inter_vol + 1e-6)

def nms_3d(centers, sizes, scores, thresh):
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        rest = []
        for j in order[1:]:
            if iou_3d(centers[i], sizes[i], centers[j], sizes[j]) < thresh:
                rest.append(j)
        order = np.array(rest)
    return keep

# =========================
# VISUALIZATION
# =========================
def create_bbox(center, size):
    l, w, h = size
    cx, cy, cz = center
    corners = np.array([
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
    ]) + center

    faces = np.hstack([
        [4,0,1,2,3],
        [4,4,5,6,7],
        [4,0,1,5,4],
        [4,1,2,6,5],
        [4,2,3,7,6],
        [4,3,0,4,7]
    ])
    return pv.PolyData(corners, faces)

# =========================
# EVALUATION
# =========================
def evaluate(dataset, net, device):
    net.eval()

    for idx in range(len(dataset)):
        pc, gt_boxes, sid = dataset[idx]

        # FIX: Ensure only xyz (3 channels)
        if pc.shape[1] > 3:
            pc = pc[:, :3]

        pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net(pc_tensor)

        scores = torch.sigmoid(pred["objectness"][0]).cpu().numpy()
        centers = pred["center"][0].cpu().numpy()
        sizes = pred["size_residual"][0].cpu().numpy() + MEAN_SIZE

        mask = scores > SCORE_THRESH
        centers, sizes, scores = centers[mask], sizes[mask], scores[mask]

        keep = nms_3d(centers, sizes, scores, NMS_IOU_THRESH)
        centers, sizes, scores = centers[keep], sizes[keep], scores[keep]

        print(f"[{sid}] detections: {len(centers)}")

        plotter = pv.Plotter()
        plotter.set_background("black")
        plotter.add_points(pc, color="white", point_size=3)

        for b in gt_boxes:
            plotter.add_mesh(create_bbox(b[:3], b[3:6]), color="green", opacity=0.4)

        for c, s, sc in zip(centers, sizes, scores):
            plotter.add_mesh(create_bbox(c, s), color="red", opacity=0.45)
            plotter.add_point_labels([c], [f"{sc:.2f}"], text_color="red")

        plotter.show()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WireVoteNetDataset(DATA_DIR)

    net = VoteNet().to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])

    evaluate(dataset, net, device)
