import enum
import torch
from torch import nn
from torch.nn import functional as F
import _global

class Net(nn.Module):
    def __init__(self, num_layer=2, act='ReLU', hidden_size=768, feature_type=0) -> None:
        super().__init__()
        self.feature_type = feature_type
        
        in_size = 768 * ([2,3,1][feature_type])
        hidden_size_in = hidden_size // 2 if act == 'GLU' else hidden_size
        Act = nn.ReLU if act == 'ReLU' else nn.GLU

        self.net = [nn.Linear(in_size, hidden_size),
                Act(),
            ] 
        for i in range(num_layer - 2):
            self.net.extend([
                nn.Linear(hidden_size_in, hidden_size),
                Act(),
            ])
        self.net.extend([nn.Linear(hidden_size_in, 2)])
        self.net = nn.Sequential(*self.net)

    def forward(self, x, y):
        if self.feature_type == 0:
            a = torch.cat((x,y), 1)
        elif self.feature_type == 1:
            a = torch.cat((x,y,x*y), 1)
        elif self.feature_type == 2:
            a = x*y
        a = self.net(a)
        return a

from models.bert import BERT

def merge(a):
    return torch.stack(a).sum(dim=0) if a!=[] else torch.zeros((1024,))

class AttnNet(nn.Module):
    def __init__(self, x_dim, y_dim, num_heads) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dim = 1024
        self.linear = nn.Linear(self.dim, x_dim * y_dim)
        self.attn = nn.MultiheadAttention(y_dim, num_heads)
        self.net = nn.Sequential(
            nn.Linear(x_dim * y_dim,self.dim),
            nn.ReLU(),
            nn.Linear(self.dim,2),
        )
        self.bert = BERT()

    def forward(self, x, y):
        _, x = self.bert(x)
        _, y = self.bert(y)
        x = torch.stack([merge(a) for a in x])
        y = torch.stack([merge(a) for a in y])
        x = x.to(_global.device)
        y = y.to(_global.device)
        # print(x.shape)
        # exit()
        x = self.linear(x).reshape(-1, self.x_dim, self.y_dim)
        y = self.linear(y).reshape(-1, self.x_dim, self.y_dim)
        x = x.permute((1,0,2))
        y = y.permute((1,0,2))
        a = self.attn(x,y,y)[0]
        y = y.permute((1,0,2))
        a = a.reshape(-1, self.x_dim * self.y_dim)
        a = self.net(a)
        return a

class ClsNet(nn.Module):
    def __init__(self, num_heads) -> None:
        super().__init__()
        self.dim = 1024
        self.cls = nn.parameter.Parameter(torch.zeros(1024),requires_grad=True)
        self.sep = nn.parameter.Parameter(torch.zeros(1024),requires_grad=True)
        self.x = nn.parameter.Parameter(torch.zeros(1024),requires_grad=True)
        self.y = nn.parameter.Parameter(torch.zeros(1024),requires_grad=True)
        self.trans = nn.TransformerEncoderLayer(self.dim, num_heads, )
        self.bert = BERT()

    def merge(self, x, y):
        batch = len(x)
        l = max([len(x[i]) + len(y[i]) for i in range(batch)])
        
        ret = torch.zeros((batch, l+3, self.dim))
        mask = torch.zeros((batch, l+3))
        for i in range(batch):
            now = [self.cls,]
            for a in x[i]:
                now.append(a + self.x)
            now.append(self.sep)
            for a in y[i]:
                now.append(a + self.y)
            now.append(self.sep)
            for j, a in enumerate(now):
                ret[i,j] = a
                mask[i,j] = 1
        ret = ret.permute((1,0,2)).to(_global.device)
        mask = mask.permute((1,0)).to(_global.device)
        return ret, mask
    
    def selfAttn(self, x, x_mask, y,  y_mask):
        x = x.permute((1,0,2))
        y = y.permute((1,0,2))
        a = self.attn(x,y,y)[0]
        y = y.permute((1,0,2))


    def forward(self, x, y):
        _, x = self.bert(x)
        _, y = self.bert(y)
        x, x_mask = self.merge(x, y)
        
        a = self.trans(x, src_key_padding_mask=x_mask)
        print(a.shape)
        exit()
        return a[0]
