import _global
import torch
import time
import random
from utils.load import AllRelation, AllSplit, Bert, Anhao

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

        bert = Bert()
        print('Bert feature LOADED.')

        self.bert = bert
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
            if len(x[1]) > 0 or len(x[2]) > 0
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
        return self.dataset.bert[a], self.dataset.bert[b], flag, (a,b,num,flag)

    def __len__(self) -> int:
        return len(self.data)

if __name__ == '__main__':
    startTime = time.process_time()
    # dataset = FeatureDataset([],part=(0, 100))
    # train = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/train.pt')
    # train = PNDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/pn/train.pt')
    dataset = AllDataset()
    train = AllSubset(dataset, 0)
    print(len(train))
    print(dataset.relation[0])
    print(train[0])
    print('the process time is: ', time.process_time() - startTime)
