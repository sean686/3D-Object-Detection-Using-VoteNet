import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils
import numpy as np

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net = net.transpose(2,1)
    batch_size, num_proposal = net.shape[0], net.shape[1]

    end_points['objectness_scores'] = net[:, :, :2]
    base_xyz = end_points['aggregated_vote_xyz']
    end_points['center'] = base_xyz + net[:,:,2:5]

    end_points['heading_scores'] = net[:,:,5:5+num_heading_bin]
    end_points['heading_residuals_normalized'] = net[:,:,5+num_heading_bin:5+num_heading_bin*2]
    end_points['heading_residuals'] = end_points['heading_residuals_normalized']*(np.pi/num_heading_bin)

    size_start = 5 + num_heading_bin*2
    end_points['size_scores'] = net[:,:,size_start:size_start+num_size_cluster]
    size_residuals = net[:,:,size_start+num_size_cluster:size_start+num_size_cluster*4].view(batch_size,num_proposal,num_size_cluster,3)
    end_points['size_residuals_normalized'] = size_residuals
    end_points['size_residuals'] = size_residuals * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    end_points['sem_cls_scores'] = net[:,:,size_start+num_size_cluster*4:]
    return end_points

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 num_proposal, sampling, seed_feat_dim=256):
        super().__init__()
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim,128,128,128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.conv1 = nn.Conv1d(128,128,1)
        self.conv2 = nn.Conv1d(128,128,1)
        self.conv3 = nn.Conv1d(128, 2+3+self.num_heading_bin*2+self.num_size_cluster*4+self.num_class,1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points):
        if self.sampling=='vote_fps':
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling=='seed_fps':
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0,end_points['seed_xyz'].shape[1],(batch_size,self.num_proposal),dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)

        end_points['aggregated_vote_xyz'] = xyz
        end_points['aggregated_vote_inds'] = sample_inds

        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)
        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return end_points
