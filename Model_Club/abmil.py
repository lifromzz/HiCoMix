import torch
from torch import nn
import torch.nn.functional as F



class ABMIL(nn.Module):
    def __init__(self, n_classes, feat_dim=1024):
        super(ABMIL, self).__init__()
        self.L = feat_dim
        self.D = 512
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L, n_classes)
        )

    def forward(self, data):
        H = data["x"]
        score = self.attention(H)  # NxK
        score = torch.transpose(score, 1, 0)  # KxN
        score_softmax = F.softmax(score, dim=1)  # softmax over N
        M = torch.mm(score_softmax, H)  # KxL
        probs = self.classifier(M)

        if not self.training:
            probs=F.softmax(probs)

        return {'score':score_softmax,'probs':probs,"z":M}


