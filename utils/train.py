import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from wire_config import WireDatasetConfig

# 🔥 PointNet++ modules
from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


# =========================
# DATASET (UNCHANGED)
# =========================
class WireVoteNetDataset(Dataset):
    def __init__(self, data_dir, num_points=20000):
        self.data_dir = data_dir
        self.num_points = num_points
        self.ids = sorted([f.replace("_pc.npz","")
                           for f in os.listdir(data_dir)
                           if f.endswith("_pc.npz")])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        pc = np.load(os.path.join(self.data_dir, f"{sid}_pc.npz"))["pc"][:, :3]

        if len(pc) >= self.num_points:
            idxs = np.random.choice(len(pc), self.num_points, replace=False)
        else:
            idxs = np.random.choice(len(pc), self.num_points, replace=True)

        pc = pc[idxs]
        boxes = np.load(os.path.join(self.data_dir, f"{sid}_bbox.npy"))

        return torch.from_numpy(pc).float(), torch.from_numpy(boxes).float()


# =========================
# MODEL (🔥 REAL VoteNet)
# =========================
class VoteNet(nn.Module):
    def __init__(self, num_proposal=256):
        super().__init__()
        self.num_proposal = num_proposal

        # ---------- PointNet++ Backbone ----------
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
        B, N, _ = pc.shape
        xyz = pc
        features = None

        # ---------- Backbone ----------
        xyz, features, _ = self.sa1(xyz, features)
        xyz, features, _ = self.sa2(xyz, features)
        xyz, features, _ = self.sa3(xyz, features)
        # xyz: (B, 512, 3)
        # features: (B, 256, 512)

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
# LOSS (UNCHANGED)
# =========================
def votenet_loss(pred, gt, cfg):
    device = pred["center"].device

    gt_center = gt[:, :, :3]
    gt_size   = gt[:, :, 3:6]

    B, P, _ = pred["center"].shape
    _, G, _ = gt_center.shape

    if G == 0:
        return torch.tensor(0.0, device=device)

    dist = torch.cdist(pred["center"], gt_center)
    min_dist, idx = dist.min(dim=2)

    obj_label = (min_dist < 0.03).float()
    obj_loss = F.binary_cross_entropy_with_logits(
        pred["objectness"], obj_label
    )

    matched_center = torch.gather(
        gt_center, 1, idx.unsqueeze(-1).expand(-1, -1, 3)
    )

    matched_size = torch.gather(
        gt_size, 1, idx.unsqueeze(-1).expand(-1, -1, 3)
    )

    mean_size = torch.tensor(
        cfg.mean_size_arr[0],
        device=device
    ).view(1,1,3)

    gt_residual = matched_size - mean_size

    center_loss = F.l1_loss(pred["center"], matched_center)
    size_loss   = F.l1_loss(pred["size_residual"], gt_residual)

    return obj_loss + center_loss + size_loss


# =========================
# COLLATE (UNCHANGED)
# =========================
def collate_fn(batch):
    pcs, boxes = zip(*batch)
    pcs = torch.stack(pcs)

    max_boxes = max(b.shape[0] for b in boxes)
    padded = []

    for b in boxes:
        if b.shape[0] < max_boxes:
            pad = torch.zeros((max_boxes - b.shape[0], 6))
            b = torch.cat([b, pad], dim=0)
        padded.append(b)

    return pcs, torch.stack(padded)


# =========================
# TRAIN
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = WireDatasetConfig()
    dataset = WireVoteNetDataset("data/votenet_format")
    loader = DataLoader(dataset, batch_size=2, shuffle=True,
                        collate_fn=collate_fn)

    net = VoteNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    os.makedirs("trained_model", exist_ok=True)

    for epoch in range(20):
        net.train()
        total = 0

        for pc, gt in loader:
            pc, gt = pc.to(device), gt.to(device)

            pred = net(pc)
            loss = votenet_loss(pred, gt, cfg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch {epoch:03d} | Loss {total/len(loader):.4f}")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, f"trained_model/votenet_epoch_{epoch}.pth")
