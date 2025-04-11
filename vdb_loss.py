import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VDBSoftmaxLoss(nn.Module):
    def __init__(self, weight, num_classes, scale=30.0, margin=0.40):
        super(VDBSoftmaxLoss, self).__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.margin = nn.Parameter(torch.tensor(margin, dtype=torch.float))
        self.weight = weight.weight
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x, labels):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        cos_theta = torch.matmul(w, x.T).T
        psi = cos_theta - self.margin
        onehot = F.one_hot(labels, self.num_classes)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)
        err = self.loss(logits, labels)
        return err
