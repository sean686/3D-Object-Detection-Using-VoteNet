# This is only if you want to skip voting for debugging
# Otherwise VoteNet uses VotingModule + ProposalModule

import torch
import torch.nn as nn
from backbone_module import Pointnet2Backbone
from proposal_module import ProposalModule

class BoxNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, sampling='vote_fps'):
        super().__init__()
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim)
        self.pnet = ProposalModule(num_class,num_heading_bin,num_size_cluster,mean_size_arr,num_proposal,sampling)

    def forward(self, inputs):
        end_points = self.backbone_net(inputs['point_clouds'])
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        end_points = self.pnet(xyz, features, end_points)
        return end_points
