import torch.nn as nn
import torch.utils.data
from Models.pointnet_backbone import *
from Common.compute_loss import *


class PointnetClassifyModel(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(PointnetClassifyModel, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.features = PointNet(
            global_feat=True, feature_transform=True, channel=channel)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.Dropout(
                p=0.4), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k),
            nn.LogSoftmax(dim=1))
        nnloss = nn.NLLLoss()

    def forward(self, x):
        x, trans, trans_feat = self.features(x)
        x = classifier(x)
        return x, trans_feat


class PointnetClassifyLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointnetClassifyLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = nnloss(pred, target)  # torch.nn.nll_loss
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
