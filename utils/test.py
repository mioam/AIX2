import _global
from utils.load import Anhao, AllRelation, Text
from utils.examples import example

from models.classifier import Net, AttnNet, ClsNet, AttnBertNet
import torch
import os
from utils.dataset import AllDataset, AllSubset
from tqdm import tqdm

@torch.no_grad()
def predict(model, sample, th=0):
    # return torch.zeros((len(sample),2))
    model.eval()
    # ret = []
    x, y, ex = [a[0] for a in sample], [a[1] for a in sample], [a[2] for a in sample]
    output = model(x,y,ex)
    ans = output.cpu()
    # for i, _ in enumerate(sample):
    #     ret.append(output[i].argmax().item())
    ans[:,0] += th
    return ans

if __name__ == '__main__':

    threshold = 0
    save_hist = False
    # model = AttnNet(96, 8 ,4)
    # path = './checkpoints/Attn.pt'
    model = AttnBertNet(96, 8 ,4)
    path = './checkpoints/AttnBert.pt'
    X = torch.load(path,map_location=_global.device)
    model.load_state_dict(X['model'])
    model.to(_global.device)

    a = Anhao()
    def check(x,y):
        if len(a[x]) == 0 or len(a[y]) == 0:
            return 0
        # print(a[x])
        s1 = set(a[x])
        s2 = set(a[y])
        if (a[y][0] == a[x][0]):
            return 3
        if (a[y][0] in s1 or a[x][0] in s2):
            return 2
        if (len(s1 | s2)):
            return 1
        return 0

    cnt = torch.zeros((2,2,4),dtype=torch.int)
    if save_hist:
        hist = [[[],[],[],[]],[[],[],[],[]]]
    dataset = AllDataset()
    # train_dataset = AllSubset(dataset, 0)
    valid_dataset = AllSubset(dataset, 1, rd=False)
    r = valid_dataset.data
    # r = AllRelation()

    text = Text()
    o1 = []
    o2 = []
    for id, p, n in tqdm(r):
        # cnt_pT = 0
        # cnt_nT = 0
        if len(p):
            pred = predict(model, [(text[id], text[x[0]], (id, x[0])) for x in p],th=threshold)
            for i, x in enumerate(p):
                label = 0
                anhao = check(id,x[0])
                # if check(id,x[0]):
                #     anhao = 1
                # else:
                #     anhao = 0
                #     o1.append((id,x[0]))
                # pred = predict(model, [(text[id],text[x[0]]), ])[0]
                # p = pred[i].argmax().item()
                p = pred[i].argmax().item() if x[1] <=1 else 1
                cnt[label,p,anhao] += 1
                if save_hist:
                    # hist[label][anhao].append(x[1])
                    hist[label][anhao].append((pred[i][1]-pred[i][0]).item())
        
        if len(n):
            pred = predict(model, [(text[id], text[x[0]], (id, x[0])) for x in n],th=threshold)
            for i, x in enumerate(n):
                label = 1
                anhao = check(id,x[0])
                # if check(id,x[0]):
                #     if len(a[id]) and len(a[x[0]]) and a[id][0] == a[x[0]][0]:
                #         anhao = 2
                #     else:
                #         anhao = 1
                #         o2.append((id,x[0]))
                # else:
                #     anhao = 0
                # p = pred[i].argmax().item()
                p = pred[i].argmax().item() if x[1] <=1 else 1
                cnt[label,p,anhao] += 1
                if save_hist:
                    # hist[label][anhao].append(x[1])
                    hist[label][anhao].append((pred[i][1]-pred[i][0]).item())

        # break
        # if cnt_nT > 10:
        #     print(cnt_pT, cnt_nT)
        #     print(a[id])
    if save_hist:
        torch.save(hist, './hist.pt')
    print(cnt)