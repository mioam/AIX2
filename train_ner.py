from random import shuffle
import torch
from torch import nn
from utils.dataset import  MRSA
from models.ner import NerNet
import _global

num_epoch = 10
lr = 0.01


net = NerNet().to(_global.device)

train = MRSA(r'./MSRA/msra_train_bio.txt')
test = MRSA(r'./MSRA/msra_test_bio.txt')

train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

optimzer = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)

loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def eva(net, test):
    test_loader = torch.utils.data.DataLoader(test, batch_size=32)
    
    s = []
    m = torch.zeros((7,7))
    for feature, label in test_loader:
        feature = feature.to(_global.device)
        label = label.to(_global.device)
        out = net(feature)
        loss = loss_fn(out,label)
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
    s, m = eva(net, test)
    print(f'epoch: {epoch}, loss: {sum(s) / len(s)}')
    print(s)
    print(m)
