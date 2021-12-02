import _global
import torch
import time
import random
from utils.load import AllRelation, AllSplit, Bert, Anhao, Text

class AllDataset:
    def __init__(self, useAnhao=False) -> None:
        relation = AllRelation()
        print('Relation LOADED.')
        if useAnhao:
            a = Anhao()
            def check(x,y):
                s1 = set(a[x])
                s2 = set(a[y])
                if s1 & s2:
                    return True
                return False
            for ele in relation:
                ele[1] = [x for x in ele[1] if check(ele[0], x[0])]
                ele[2] = [x for x in ele[2] if check(ele[0], x[0])]
        split = AllSplit()
        print('Split LOADED.')
        print(len(split), len([0 for x in split if x == 0]))
        # print(len(relation))
        # print(split[0])
        # print(relation[0:10])
        # exit()

        text = Text()
        print('text LOADED.')

        self.text = text
        self.split = split
        self.relation = relation


class AllSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, part, rd=True) -> None:
        super().__init__()
        self.dataset = dataset
        self.part = part
        self.rand = rd
        data = [
            [x[0], x[1], [a for a in x[2] if self.dataset.split[a[0]] == self.part]]
            for x in self.dataset.relation
            if self.dataset.split[x[0]] == self.part
        ]
        self.data = [
            x for x in data
            if len(x[1]) > 0 and len(x[2]) > 0
        ]
        self.docs = [i for i,x in enumerate(self.dataset.split) if x == part]
        # print(self.dataset.split[self.dataset.relation[0][0]])
    
    def __getitem__(self, index):
        x = self.data[index]
        a = x[0]
        r1 = random.random()
        r2 = random.random()

        # k = (len(x[2]) + 0.5) / (len(x[1]) + len(x[2]) + 1)
        k = 0.5

        if self.rand and (r1 < k or len(x[1]) == 0) and r2 < 0.1:
            b = random.sample(self.docs,1)[0]
            num = -1
            flag = 1
        elif (r1 < k or len(x[1]) == 0) and len(x[2]) > 0:
            b, num = random.sample(x[2],1)[0]
            flag = 1
        else:
            b, num = random.sample(x[1],1)[0]
            flag = 0

        # if len(x[1]) >= 0 and (r1 < 0.5 or r2 < 0.5 and len(x[2]) == 0) and :
        #     # if len(x[1]) == 0:
        #     #     print(x)
        #     b, num = random.sample(x[1],1)[0]
        #     flag = 0
        # elif len(x[2]) > 0:
        # print(a,b)
        return self.dataset.text[a], self.dataset.text[b], flag, (a,b,num,flag)

    def __len__(self) -> int:
        return len(self.data)

from models.bert import BERT

class MRSA(torch.utils.data.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        bert = BERT()
        data = []
        label = []
        self.d = {'O':0, 'B-LOC':1, 'I-LOC':2, 'B-ORG':3, 'I-ORG':4, 'B-PER':5, 'I-PER':6}
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                # print(line)
                if len(line) >= 4:
                    data.append(line[0])
                    label.append(self.getLabel(line[2:-1]))
        print(len(data))
        self.l = 500
        self.len = len(data) // self.l
        self.feature = []
        self.label = []
        for i in range(self.len):
            now = data[i*self.l: (i+1)*self.l]
            now_label = label[i*self.l: (i+1)*self.l]
            out, tokens = bert.getBert(''.join(now))
            # if out.shape[1] != len(now)+2:
            #     print(now)
            #     print(out.shape)
            for j in range(len(now_label)):
                pos = tokens.char_to_token(j)
                self.feature.append(out[0,pos])
                self.label.append(torch.tensor(now_label[j]))

    def getLabel(self, x):
        return self.d[x]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return self.feature[i], self.label[i]
        

if __name__ == '__main__':
    startTime = time.process_time()
    mrsa = MRSA(r'C:\something\AI+X\project\datasets\MSRA\msra_train_bio.txt')
    print(mrsa[0])
    exit()

    # dataset = FeatureDataset([],part=(0, 100))
    # train = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/train.pt')
    # train = PNDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/pn/train.pt')
    dataset = AllDataset(useAnhao=True)
    train = AllSubset(dataset, 0)
    print(len(train))
    print(dataset.relation[0])
    # print(sum([x[1] for x in train.data]))
    # print(sum([x[2] for x in train.data]))
    from utils.load import Text, Html, Anhao
    content = Html()
    anhao = Anhao()
    l = [(x[0], a[0]) for x in train.data for a in x[2]]
    print(len(l))
    l = random.sample(l, 20)
    for i, (x,y) in enumerate(l):
        # print(i)
        with open(f'_SameAnhao/{i}_x.html','w',encoding='utf8') as f:
            f.write(content[x])
        with open(f'_SameAnhao/{i}_y.html','w',encoding='utf8') as f:
            f.write(content[y])
        with open(f'_SameAnhao/{i}_z.txt','w',encoding='utf8') as f:
            f.write(str(set(anhao[x]) & set(anhao[y])))
    # print(train[0])
    print('the process time is: ', time.process_time() - startTime)

