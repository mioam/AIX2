import _global
from utils.load import Anhao, AllRelation, Text
from utils.examples import example

from models.classifier import Net, AttnNet, ClsNet
import torch
import os
from tqdm import tqdm

@torch.no_grad()
def predict(model, relation):
    model.eval()
    ret = []
    for x, y in relation:
        output = model([x],[y])
        # print(output)
        ret.append(output[0].argmax().item())
    return ret

if __name__ == '__main__':

    model = AttnNet(96, 8 ,4)
    path = './checkpoint/Attn.pt'
    X = torch.load(path,map_location=_global.device)
    model.load_state_dict(X['model'])
    model.to(_global.device)

    a = Anhao()
    def check(x,y):
        if len(a[x]) == 0 or len(a[y]) == 0:
            return False
        # print(a[x])
        s1 = set(a[x])
        s2 = set(a[y])
        if (a[y][0] in s1 or a[x][0] in s2):
            return True
        return False

    cnt = torch.zeros((2,3,3))
    r = AllRelation()
    text = Text()
    o1 = []
    o2 = []
    for id, p, n in tqdm(r):
        
        # cnt_pT = 0
        # cnt_nT = 0
        for x in p:
            label = 0
            if check(id,x[0]):
                anhao = 1
            else:
                anhao = 0
                o1.append((id,x[0]))
            pred = predict(model, [(text[id],text[x[0]]), ])[0]
            cnt[label,pred,anhao] += 1
        for x in n:
            label = 1
            if check(id,x[0]):
                if len(a[id]) and len(a[x[0]]) and a[id][0] == a[x[0]][0]:
                    anhao = 2
                else:
                    anhao = 1
                    o2.append((id,x[0]))
            else:
                anhao = 0
            pred = predict(model, [(text[id],text[x[0]]), ])[0]
            cnt[label,pred,anhao] += 1

        # if cnt_nT > 10:
        #     print(cnt_pT, cnt_nT)
        #     print(a[id])
    
    print(cnt)