import torch
import torch.nn as nn
import torch.nn.functional as F

def get_act(act):
    if act.lower() == 'relu':
        return nn.ReLU()
    elif act.lower() == 'gelu':
        return nn.GELU()
    elif act.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif act.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif act.lower() == 'tanh':
        return nn.Tanh()
    elif act.lower() == 'silu':
        return nn.SiLU()
    else:
        raise Exception(f'Invalid activation function: {act}')

class Attention(nn.Module):
    def __init__(self,in_dim=512,hid_dim=512, out_dim=128,act='relu',bias=False,dropout=0):
        super(Attention, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.L = hid_dim
        self.D = out_dim #128
        self.K = 1

        self.feature = [nn.Linear(in_dim, hid_dim)]
        self.feature += [get_act(act)]
        self.feature += [nn.Dropout(self.dropout)]
        self.feature = nn.Sequential(*self.feature)

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu':
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)


    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        return x, A
