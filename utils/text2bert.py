import _global
import torch
from transformers import AutoTokenizer, AutoModel

# DEVICE = _global.device

class BERT:
    def __init__(self):
        self.device = _global.device
        name = r'hfl/chinese-legal-electra-base-discriminator'
        # name = 'hfl/chinese-roberta-wwm-ext'
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.bert = AutoModel.from_pretrained(name)
        self.bert.to(self.device)
    @torch.no_grad()
    def __call__(self, x):
        # print(self.HanLP(x))
        # exit()
        # print(x)
        tokens = self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
        out = self.bert(**tokens)
        print(out['last_hidden_state'].shape)
        print(out['pooler_output'].shape)
        return out['last_hidden_state'].cpu()

def get_bert(bert, text_arr):
    if text_arr == []:
        return []
    result = bert(text_arr)
    return result

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
    # print(a)
    bert = BERT()
    bert_arr = []
    for t in range(0,len(a),512):
        last_hidden_state = get_bert(bert, a[t:t+512])
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
    x = load.Text()[:10]
    a = getBERT(x)
    print(a)
    print(len(a))
    print(len(a[0]))
    print(a[0][0].shape)
    print(a[0][1].shape)