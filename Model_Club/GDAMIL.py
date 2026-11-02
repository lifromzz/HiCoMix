import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GlobalAttention

from Model_Club.Attention import Attention

import warnings
warnings.filterwarnings("ignore")


class GDAMIL(nn.Module):

    def __init__(self, in_dim=1024, num_classes=2, hid_dim=256, out_dim=128, k_components=10, k_neighbors=10,
                 dropout=0.1, lambda_smooth=0.0, lambda_nce=0.0):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.LeakyReLU())
        self.bag_partition = BagPartition(hid_dim, hid_dim, out_dim, k_components)
        self.gnn = GraphAttention(hid_dim, in_channels=out_dim, k_neighbors=k_neighbors)
        self.basic = Attention(hid_dim, hid_dim, out_dim, dropout=dropout)
        self.feature_fusion = FeatureFusion(dim=out_dim)
        self.bag_embedding = GlobalAttention(
            nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(out_dim // 2, 1)
            )
        )
        self.lambda_smooth = lambda_smooth
        self.lambda_nce = lambda_nce

        self.basic_linear = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.LeakyReLU())
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_classes)
        )

    def forward(self, data):
        x = data["x"]
        X = self.encoder(x)
        X_soft = self.bag_partition(X)

        X_soft, edge_index_soft = self.gnn(X_soft)  # k_components x 128

        b_gnn = self.bag_embedding(X_soft)  # 1 x 128
        b_basic, A = self.basic(X)
        b_basic = self.basic_linear(b_basic)

        b = self.feature_fusion(b_gnn, b_basic)

        logits = self.classifier(b)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs = logits
        if not self.training:
            probs = F.softmax(probs, dim=-1)

        z = b.squeeze(0) if b.dim() > 1 else b
        return {"probs": probs, "score": A, "z": z}

class BagPartition(nn.Module):

    def __init__(self, in_dim=512, hid_dim=512, out_dim=128, k_components=10):
        super().__init__()

        self.k_components = k_components
        self.attention = Attention(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)
        self.cluster_logits = nn.Linear(in_dim, k_components)

    def forward(self, X):
        X = X.squeeze(0)
        assert len(X.shape) == 2
        P = F.gumbel_softmax(self.cluster_logits(X), tau=0.5, hard=False)
        partitions = P.T @ X  # n x 512

        return partitions

class DynamicGraphBuilder(nn.Module):

    def __init__(self, dim, topk=10):
        super().__init__()
        self.W_head = nn.Linear(dim, dim)
        self.W_tail = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.topk = topk

    def forward(self, X):
        # n: The number of instances in the new bag
        e_h = self.W_head(X)  # n x 512
        e_t = self.W_tail(X)  # n x 512
        logits = (e_h @ e_t.T) * self.scale  # n x n

        # topk for each row
        topk_val, topk_idx = torch.topk(logits, k=self.topk, dim=-1)
        weights = F.softmax(topk_val, dim=-1)  # n' * topk

        # Construct the edge_index and edge_weight
        n = X.size(0)
        src = torch.arange(n).unsqueeze(1).expand(-1, self.topk).reshape(-1).to(X.device)
        dst = topk_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = weights.reshape(-1)
        return edge_index, edge_weight


class GraphAttention(nn.Module):

    def __init__(self, in_dim, in_channels, num_layers=1, k_neighbors=10):
        super().__init__()

        self.convs_list = nn.ModuleList()
        self.convs_list.append(SAGEConv(in_channels=in_dim, out_channels=in_channels))
        self.graph_builder = DynamicGraphBuilder(in_dim, topk=k_neighbors)
        # self.graph_builder = SparseGraphBuilder(k=k_neighbors)

        for _ in range(num_layers - 1):
            self.convs_list.append(SAGEConv(in_channels=in_channels, out_channels=in_channels))

        self.lin_sum = nn.Linear(in_channels, in_channels)
        self.lin_bi = nn.Linear(in_channels, in_channels)
        self.gate_U = nn.Linear(in_channels, in_channels // 2)
        self.gate_V = nn.Linear(in_channels, in_channels // 2)
        self.gate_W = nn.Linear(in_channels // 2, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, X):

        edge_index, edge_weight = self.graph_builder(X)

        # adj = self.graph_builder.build(X)
        # edge_index, edge_value = adj.indices(), adj.values()
        for conv in self.convs_list:
            X = F.leaky_relu(conv(X, edge_index))

        row, col = edge_index
        # summed = Σ_j w_ij * X[j]
        summed = torch.zeros_like(X).index_add_(0, row, X[col] * edge_weight.unsqueeze(1))  # n x 128
        sum_msg = self.lin_sum(X + summed)
        bi_msg = self.lin_bi(X * summed)

        u = self.gate_U(X)  # n x 64
        v = self.gate_V(summed)  # n x 64
        g = torch.sigmoid(self.gate_W(u + v))  # n x 128

        out = F.leaky_relu(g * sum_msg + (1 - g) * bi_msg)  # n x 128
        out = self.norm(out + X)

        return out, edge_index


class FeatureFusion(nn.Module):
    def __init__(self, dim=32):
        super(FeatureFusion, self).__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU()
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        gate = self.gate(combined)
        transformed = self.transform(combined)
        return gate * x1 + (1 - gate) * transformed


# class AdaptiveBagPartition(nn.Module):
#     """自适应包分区，解决信息丢失和k值选择问题"""
#
#     def __init__(self, in_dim=512, hid_dim=256, out_dim=128,
#                  min_components=5, max_components=20):
#         super().__init__()
#
#         self.min_components = min_components
#         self.max_components = max_components
#
#         # The adaptive predictor for the number of components
#         self.component_predictor = nn.Sequential(
#             nn.Linear(in_dim, hid_dim // 2),
#             nn.LeakyReLU(),
#             nn.Linear(hid_dim // 2, 1),
#             nn.Sigmoid()
#         )
#
#         self.multi_scale_conv = nn.ModuleList([
#             nn.Conv1d(in_dim, out_dim, kernel_size=1),
#             nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
#             nn.Conv1d(in_dim, out_dim, kernel_size=5, padding=2)
#         ])
#
#         self.info_preserve = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.LayerNorm(out_dim)
#         )
#
#         self.soft_cluster = nn.Linear(in_dim, max_components)
#
#         self.residual_gate = nn.Sequential(
#             nn.Linear(out_dim * 2, out_dim),
#             nn.Sigmoid()
#         )

