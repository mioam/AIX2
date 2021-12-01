import torch
from torch import nn
from torch.nn import functional as F
import _global
from transformers import AutoTokenizer, AutoModel


class BERT:
    def __init__(self):
        self.device = _global.device
        # name = r'hfl/chinese-legal-electra-base-discriminator'
        # name = 'hfl/chinese-roberta-wwm-ext'
        self.tokenizer = AutoTokenizer.from_pretrained(_global.electra_g)
        self.bert = AutoModel.from_pretrained(_global.electra_d)
        self.bert = nn.DataParallel(self.bert ,device_ids=[0,1,2,3,4,5,6,7])
        self.bert.to(self.device)
    @torch.no_grad()
    def getBert(self, x):
        tokens = self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
        out = self.bert(**tokens)
        return out['last_hidden_state'].cpu().detach()

    def __call__(self, texts):
        # print(texts)
        # exit()
        a = []
        lines = []
        for text in texts:
            l = len(a)
            for i in range(len(text)):
                tmp = text[i]
                if i:
                    tmp = text[i-1] + tmp
                if i != len(text) - 1:
                    tmp = tmp + text[i+1]
                a.append(tmp)
            r = len(a)
            lines.append([l, r])
        # print('ok')
        bert_arr = []
        batchsize = 32 * 8
        for t in range(0,len(a),batchsize):
            last_hidden_state = self.getBert(a[t:t+batchsize])
            # print(last_hidden_state)
            # exit()
            last_hidden_state = [last_hidden_state[i] for i in range(last_hidden_state.shape[0])]
            bert_arr.extend(last_hidden_state)
        # print(key_arr)
        # return
        # print('okk') 
        ret = []
        cls = []
        for (l, r), text in zip(lines, texts):
            now = bert_arr[l:r]
            bert_ret = []
            bert_cls = []
            for i in range(len(text)):
                start = 1
                end = len(text[i]) + 1
                if i:
                    start += len(text[i-1])
                    end += len(text[i-1])
                tmp = now[i][start:end]
                bert_ret.append(tmp)
                bert_cls.append(now[i][0])
            
            ret.append(bert_ret)
            cls.append(bert_cls)
        # print('okkkk') 
        return ret, cls