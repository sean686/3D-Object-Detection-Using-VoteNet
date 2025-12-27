import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WireVoteNetDataset(Dataset):
    def __init__(self, data_dir, num_points=20000, max_num_obj=6):
        self.data_dir = data_dir
        self.num_points = num_points
        self.max_num_obj = max_num_obj

        self.pc_files = sorted(f for f in os.listdir(data_dir) if f.endswith("_pc.npz"))

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        scene_id = self.pc_files[idx].replace("_pc.npz","")

        pc = np.load(f"{self.data_dir}/{scene_id}_pc.npz")['pc']
        votes = np.load(f"{self.data_dir}/{scene_id}_votes.npz")
        boxes = np.load(f"{self.data_dir}/{scene_id}_bbox.npy")

        point_votes = votes['point_votes']
        vote_mask = votes['vote_mask']

        K = min(len(boxes), self.max_num_obj)

        center_label = np.zeros((self.max_num_obj,3), dtype=np.float32)
        size_class_label = np.zeros((self.max_num_obj,), dtype=np.int64)
        size_residual_label = np.zeros((self.max_num_obj,3), dtype=np.float32)

        for i in range(K):
            center_label[i] = boxes[i,0:3]
            size_residual_label[i] = boxes[i,3:6]

        return {
            'point_clouds': torch.from_numpy(pc).float(),
            'vote_label': torch.from_numpy(point_votes).float(),
            'vote_mask': torch.from_numpy(vote_mask).long(),
            'center_label': torch.from_numpy(center_label).float(),
            'size_class_label': torch.from_numpy(size_class_label).long(),
            'size_residual_label': torch.from_numpy(size_residual_label).float(),
            'num_gt_objects': torch.tensor(K).long()
        }
