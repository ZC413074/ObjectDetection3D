import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()
        layers = []
        input_dim, output_dim = channel, 64
        for i in range(3):
            layers += [nn.Conv1d(input_dim, output_dim, 1),
                       nn.BatchNorm1d(output_dim), nn.ReLU()]  # inplace=True
            input_dim = output_dim
            output_dim = input_dim*2 if i < 2 else input_dim*8
        layers += nn.MaxPool1d(1)
        output_dim = input_dim*0.5
        for i in range(3):
            if i < 2:
                layers += [nn.Linear(input_dim, output_dim, 1),
                           nn.BatchNorm1d(output_dim), nn.ReLU()]  # inplace=True
            elif i == 2:
                layers += [nn.Linear(input_dim, output_dim, 1)]  # inplace=True
            input_dim = output_dim
            output_dim = input_dim*0.5
        self.tnet = nn.Sequential(*layers)
        self.k = k
    def forward(self, x):
        batchsize = x.size()[0]
        x = self.tnet(x)
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.tnet3 = TNet(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.tnet64 = TNet(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.tnet3(x, D)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.tnet64(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
