from torch import nn
import torch
import torch.nn.functional as F

class Feat_Projecter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024):
        super(Feat_Projecter, self).__init__()
        self.projecter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x = [B, N, C] or [N, C]
        if len(x.shape) == 3:
            L1, L2, L3 = x.shape[0], x.shape[1], x.shape[2]
            x = x.view(-1, L3)
            x = self.projecter(x) # for BatchNorm
            x = x.view(L1, L2, -1)
        else:
            x = self.projecter(x)
        return x


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, hid_size, output_class, dropout_v=0.0):  # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, hid_size)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, hid_size)
        )

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=hid_size)

    def forward(self, feats, c, **kwargs):  # N x K, N x C
        # print(feats.shape[0])
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
                      0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(nn.Module):
    def __init__(self, dim_in=1024, dim_emb=1024, num_cls=2, use_feat_proj=True, drop_rate=0.5):
        super(DSMIL, self).__init__()
        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
        self.i_classifier = FCLayer(in_size=dim_in, out_size=num_cls)
        self.b_classifier = BClassifier(dim_in, dim_emb, num_cls, dropout_v=drop_rate)

    def forward(self, data):
        H = data["x"]
        # print(data["x"].size())
        X = self.feat_proj(H)
        feats, classes = self.i_classifier(X)

        prediction_bag, A, B = self.b_classifier(feats, classes)  # bag = [1, C], A = [N, C]

        max_prediction, _ = torch.max(classes, 0)
        logits = 0.5 * (prediction_bag + max_prediction)  # logits = [1, C]

        # if "y_a" in data.keys():
        #     loss=self.compute_loss(max_prediction,prediction_bag,data[""])
        if not self.training:
            logits=F.softmax(logits)
        return {"probs":logits,"score":A,"classes":classes,"z":B.mean(dim=1)}

    def compute_loss(self,max_prediction,prediction_bag,y):
        loss_fn=nn.BCEWithLogitsLoss()
        loss=0.5*(loss_fn(max_prediction.view(1,-1),y.view(1,-1))+loss_fn(prediction_bag.view(1,-1),y.view(1,-1)))
        return loss

if __name__ == "__main__":
    model = DSMIL()
    x = torch.randn((1600, 1024))
    result= model({"x":x})
