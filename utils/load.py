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
    t = Text()
    # # print(len(x))
    # id = [i for i, aa in enumerate(a) if len(aa) == 0 ]
    # print(len(id))
    # for j in range(20):
    #     print(t[id[j]])
    # print(h[id[0]])

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

    cnt_p = 0
    cnt_pT = 0
    cnt_n = 0
    cnt_nT = 0
    cnt_np = 0 
    r = AllRelation()
    o1 = []
    o2 = []
    for id, p, n in r:
        # cnt_pT = 0
        # cnt_nT = 0
        for x in p:
            cnt_p += 1
            if check(id,x[0]):
                cnt_pT +=1
            else:
                o1.append((id,x[0]))
        for x in n:
            cnt_n += 1
            if check(id,x[0]):
                if len(a[id]) and len(a[x[0]]) and a[id][0] == a[x[0]][0]:
                    cnt_np += 1
                else:
                    cnt_nT +=1
                    o2.append((id,x[0]))

        # if cnt_nT > 10:
        #     print(cnt_pT, cnt_nT)
        #     print(a[id])
    
    print(f'数据集中有标号的对数: {cnt_p}, 其中这个方法找到的有标号的对数: {cnt_pT},\n 数据集中无标号的对数: {cnt_n}, 其中这个方法以为有标号的对数: {cnt_nT + cnt_np}, 其中有可能是漏标号的对数: {cnt_np}')

    from utils.examples import example

    example(o1, '_positive_but_no_anhao')
    example(o2, '_negative_but_same_anhao', same_anhao=True)