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
        self.trans = nn.TransformerEncoderLayer(self.dim, num_heads, dim_feedforward=128)
        self.bert = BERT()
        self.linear = nn.Linear(self.dim, 2)
        self.cache = {}

    def merge(self, x, y):
        batch = len(x)
        l = max([len(x[i]) + len(y[i]) for i in range(batch)])+1
        
        ret = torch.zeros((batch, l+3, self.dim))
        tmp = []
        mask = torch.zeros((batch, l+3),dtype=torch.bool)
        for i in range(batch):
            now = [torch.zeros((self.dim)),]
            now_tmp = [self.cls,]
            for a in x[i]:
                now.append(a)
                now_tmp.append(self.x)
            now.append(torch.zeros((self.dim)))
            now_tmp.append(self.sep)
            for a in y[i]:
                now.append(a)
                now_tmp.append(self.y)
            now.append(torch.zeros((self.dim)))
            now_tmp.append(self.sep)

            now_tmp = torch.stack(now_tmp) # l * 1024
            now_tmp = F.pad(now_tmp,(0, 0, 0, l+3-now_tmp.shape[0]))
            tmp.append(now_tmp)
            for j, a in enumerate(now):
                ret[i,j] = a
                mask[i,j] = True
        # print(ret)
        ret = ret.to(_global.device) + torch.stack(tmp).to(_global.device)
        # print(ret)
        ret = ret.permute((1,0,2))
        mask = mask.to(_global.device)
        return ret, mask

    def forward(self, x, y, ex=None):
        if ex is not None and ex[0] in self.cache:
            x = self.cache[ex[0]]
        else:
            _, x = self.bert(x)
            self.cache[ex[0]] = x
        if ex is not None and ex[1] in self.cache:
            y = self.cache[ex[1]]
        else:
            _, y = self.bert(y)
            self.cache[ex[1]] = y

        # print(x)
        x, x_mask = self.merge(x, y)
        # print(x, x_mask)
        # print(x_mask.shape, x_mask[:,-1])
        
        a = self.trans(x, src_key_padding_mask=x_mask)
        # print(a[0])
        # exit()
        return self.linear(a[0])
