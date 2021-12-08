import enum
import torch
from torch import nn
from torch.nn import functional as F
from utils import load
import _global

from models.ner import NerNet

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


class AttnBertNet(nn.Module):
    def __init__(self, x_dim, y_dim, num_heads) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dim = 768
        self.linear = nn.Linear(self.dim, x_dim * y_dim)
        self.attn = nn.MultiheadAttention(y_dim, num_heads)
        self.net = nn.Sequential(
            nn.Linear(x_dim * y_dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim,2),
        )
        self.a = load.Bert()

    def forward(self, x, y, ex):
        # print(ex)
        x = []
        y = []
        for e in ex:
            x.append(self.a[e[0]])
            y.append(self.a[e[1]])
        x = torch.stack(x).to(_global.device)
        y = torch.stack(y).to(_global.device)
        # print(x.shape)
        x = self.linear(x).reshape(-1, self.x_dim, self.y_dim)
        y = self.linear(y).reshape(-1, self.x_dim, self.y_dim)
        x = x.permute((1,0,2))
        y = y.permute((1,0,2))
        a = self.attn(x,y,y)[0]
        a = a.permute((1,0,2))
        a = a.reshape(-1, self.x_dim * self.y_dim)
        a = self.net(a)
        return a

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

    def forward(self, x, y, ex=None):
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
        a = a.permute((1,0,2))
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
        if ex is not None:
            x0 = []
            y0 = []
            for _x, _y, e in zip(x,y,ex):
                if e[0] not in self.cache:
                    _, a = self.bert([_x])
                    a = a[0]
                    self.cache[e[0]] = a
                x0.append(self.cache[e[0]])
                
                if e[1] not in self.cache:
                    _, a = self.bert([_y])
                    a = a[0]
                    self.cache[e[1]] = a
                y0.append(self.cache[e[1]])
            x = x0
            y = y0
        else:
            _, x = self.bert(x)
            _, y = self.bert(y)

        # print(x)
        x, x_mask = self.merge(x, y)
        # print(x, x_mask)
        # print(x_mask.shape, x_mask[:,-1])
        
        a = self.trans(x, src_key_padding_mask=x_mask)
        # print(a[0])
        # exit()
        return self.linear(a[0])

class KeyNet(nn.Module):
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

        self.ner = NerNet()
        self.ner.load_state_dict(torch.load('./checkpoints/ner.pt',map_location='cpu'))
        self.ner.to(_global.device)
        self.ner.eval()

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
    
    @torch.no_grad()
    def getkey(self, x):
        # print(x)
        # print(len(x))
        ret = []
        for e in x:
            # print(len(e))
            now = []
            for a in e:
                # print(a.shape)
                out = self.ner(a.to(_global.device)).cpu()
                for key in range(out.shape[0]):
                    if out[key].argmax() == 5:
                        now.append(a[key])
            ret.append(now[:10])
        return ret

    def forward(self, x, y, ex=None):
        if ex is not None:
            x = []
            y = []
            for e in ex:
                if e[0] not in self.cache:
                    _, a = self.bert([x])
                    a = self.getkey(a)
                    a = a[0]
                    self.cache[e[0]] = a
                x.append(self.cache[e[0]])
                
                if e[1] not in self.cache:
                    _, a = self.bert([y])
                    a = self.getkey(a)
                    a = a[0]
                    self.cache[e[1]] = a
                y.append(self.cache[e[1]])
        else:
            _, x = self.bert(x)
            _, y = self.bert(y)
            x = self.getkey(x)
            y = self.getkey(y)
        
        # if ex is not None and ex[0] in self.cache:
        #     x = self.cache[ex[0]]
        # else:
        #     x, _ = self.bert(x)
        #     x = self.getkey(x)
        #     self.cache[ex[0]] = x
        
        # if ex is not None and ex[1] in self.cache:
        #     y = self.cache[ex[1]]
        # else:
        #     y, _ = self.bert(y)
        #     y = self.getkey(y)
        #     self.cache[ex[1]] = y

        # print(x)
        x, x_mask = self.merge(x, y)
        # print(x, x_mask)
        # print(x_mask.shape, x_mask[:,-1])
        
        a = self.trans(x, src_key_padding_mask=x_mask)
        # print(a[0])
        # exit()
        return self.linear(a[0])