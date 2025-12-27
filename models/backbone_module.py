import torch
import torch.nn as nn
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class Pointnet2Backbone(nn.Module):
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048, radius=0.2, nsample=64,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True, normalize_xyz=True
        )
        self.sa2 = PointnetSAModuleVotes(
            npoint=1024, radius=0.4, nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True, normalize_xyz=True
        )
        self.sa3 = PointnetSAModuleVotes(
            npoint=512, radius=0.8, nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True, normalize_xyz=True
        )
        self.sa4 = PointnetSAModuleVotes(
            npoint=256, radius=1.2, nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True, normalize_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[512, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[512, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: torch.Tensor, end_points=None):
        if end_points is None: end_points = {}
        xyz, features = self._break_up_pc(pointcloud)

        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_xyz'], end_points['sa1_features'], end_points['sa1_inds'] = xyz, features, fps_inds

        xyz, features, fps_inds = self.sa2(xyz, features)
        end_points['sa2_xyz'], end_points['sa2_features'], end_points['sa2_inds'] = xyz, features, fps_inds

        xyz, features, fps_inds = self.sa3(xyz, features)
        end_points['sa3_xyz'], end_points['sa3_features'] = xyz, features

        xyz, features, fps_inds = self.sa4(xyz, features)
        end_points['sa4_xyz'], end_points['sa4_features'] = xyz, features

        # Feature upsampling
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)

        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        end_points['fp2_inds'] = end_points['sa1_inds'][:, :end_points['fp2_xyz'].shape[1]]
        return end_points
