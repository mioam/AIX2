import _global
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
# DEVICE = _global.device

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
    def __call__(self, x):
        # print(self.HanLP(x))
        # exit()
        # print(x)
        tokens = self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
        out = self.bert(**tokens)
        # print(out)
        # print(len(x), x)
        # print(out['last_hidden_state'].shape)
        # print(out['pooler_output'].shape)
        return out['last_hidden_state'].cpu().detach()

def get_bert(bert, text_arr):
    if text_arr == []:
        return []
    result = bert(text_arr)
    return result

@torch.no_grad()
def getBERT(texts):
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
    del texts
    # print(a)
    bert = BERT()
    bert_arr = []
    batchsize = 32 * 8
    for t in tqdm(range(0,len(a),batchsize)):
        last_hidden_state = get_bert(bert, a[t:t+batchsize])
        # print(last_hidden_state)
        # exit()
        last_hidden_state = [last_hidden_state[i] for i in range(last_hidden_state.shape[0])]
        bert_arr.extend(last_hidden_state)
    # print(key_arr)
    # return 
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
    return ret, cls
    # ans = []
    # tokenizer = BertTokenizerFast.from_pretrained(
    #     'hfl/chinese-roberta-wwm-ext')
    # bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # bert.to(DEVICE)
    # for text in texts:
    #     cls = torch.zeros(768)
    #     cnt_cls = 0
    #     outputs = torch.zeros((len(text),768))
    #     cnt_outputs = torch.zeros((len(text),1))
    #     for i in range(0, len(text), STEP):
    #         s = text[i:i+WINDOW]
    #         tokens = tokenizer(s, return_tensors="pt", padding=True).to(DEVICE)
    #         out = bert(**tokens)
    #         cls += out.last_hidden_state[0,0].cpu()
    #         cnt_cls += 1
    #         outputs[i:i+WINDOW] += out.last_hidden_state[0,1:-1].cpu()
    #         cnt_outputs[i:i+WINDOW] += 1
    #     if cnt_cls:
    #         cls = cls / cnt_cls
    #     outputs = outputs / cnt_outputs
    #     ans.append((cls, outputs))
    # return ans

if __name__ == '__main__':
    # a = getBERT([['你好。', '阿斯顿法国红酒看来。']])

    import utils.load as load
    x = load.Text()[:2]
    ret, cls = getBERT(x)
    # print(a)
    print(len(ret))
    print(len(ret[0]))
    print(ret[0][0].shape)

    print(len(cls))
    print(len(cls[0]))
    print(cls[0][0].shape)
