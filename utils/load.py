import _global
import os
import torch

def Bert(path=_global.bertPath):
    bert = torch.load(path)
    return bert

def Html(path=_global.htmlPath):
    html = torch.load(path)
    return html

def Text(path=_global.textPath):
    text = torch.load(path)
    return text

def Anhao(path=_global.anhaoPath):
    anhao = torch.load(path)
    return anhao

def AllRelation(path=_global.allRelationPath):
    relation = torch.load(path)
    return relation

def AllSplit(path=_global.allSplitPath):
    split = torch.load(path)
    return split

if __name__ == '__main__':
    # h = Html()
    # a = Anhao()
    # t = Text()
    # print(len(x))
    # id = [i for i, a in enumerate(x) if len(a) == 0 ]
    # print(len(id))
    # for j in range(20):
    #     print(t[id[j]])
    # # print(h[id[0]])

    a = Anhao()
    def check(x,y):
        # print(a[x])
        s1 = set(a[x])
        s2 = set(a[y])
        if s1 & s2:
            return True
        return False

    cnt_p = 0
    cnt_pT = 0
    cnt_n = 0
    cnt_nT = 0
    r = AllRelation()
    for id, p, n in r:
        for x in p:
            cnt_p += 1
            if check(id,x[0]):
                cnt_pT +=1
        for x in n:
            cnt_n += 1
            if check(id,x[0]):
                cnt_nT +=1

    
    print(cnt_p,
    cnt_pT,
    cnt_n,
    cnt_nT)