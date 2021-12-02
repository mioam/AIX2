from random import shuffle
import torch
from torch import nn
from utils.dataset import  MRSA
from models.ner import NerNet
import _global

num_epoch = 100
lr = 1e-3


net = NerNet().to(_global.device)

train = MRSA(r'./MSRA/msra_train_bio.txt', r'./_train.pt')
test = MRSA(r'./MSRA/msra_test_bio.txt', r'./_test.pt')

train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

optimzer = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)

loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def eva(net, test):
    net.eval()
    test_loader = torch.utils.data.DataLoader(test, batch_size=32)
    
    s = []
    m = torch.zeros((7,7),dtype=torch.int)
    for feature, label in test_loader:
        feature = feature.to(_global.device)
        label = label.to(_global.device)
        out = net(feature)
        loss = loss_fn(out, label)
        s.append(loss.item())
        for i in range(out.shape[0]):
            a = out[i].argmax().item()
            b = label[i].item()
            m[a,b] += 1
    return sum(s) / len(s), m

for epoch in range(num_epoch):
    net.train()
    s = []
    for feature, label in train_loader:
        feature = feature.to(_global.device)
        label = label.to(_global.device)
        out = net(feature)
        loss = loss_fn(out,label)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        # print(loss.item())
        s.append(loss.item())
    print(f'epoch: {epoch}, loss: {sum(s) / len(s)}')
    sv, mv = eva(net, test)
    print(sv)
    print(mv)
    for i in range(7):
        tp = mv[i,i]
        fp = mv[i,:].sum() - tp
        fn = mv[:,i].sum() - tp
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        F1  = 2 * recall * precision / (recall + precision)
        print(f'{i}: recall {recall}, precision: {precision}, F1: {F1}')

torch.save(net.state_dict(),'./checkpoints/ner.pt')