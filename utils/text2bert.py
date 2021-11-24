import _global
import torch
from transformers import BertTokenizerFast, BertModel

STEP = 400
WINDOW = 500
DEVICE = _global.device
def getBERT(texts):
    ans = []
    tokenizer = BertTokenizerFast.from_pretrained(
        'hfl/chinese-roberta-wwm-ext')
    bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
    bert.to(DEVICE)
    for text in texts:
        cls = torch.zeros(768)
        cnt_cls = 0
        outputs = torch.zeros((len(text),768))
        cnt_outputs = torch.zeros((len(text),1))
        for i in range(0, len(text), STEP):
            s = text[i:i+WINDOW]
            tokens = tokenizer(s, return_tensors="pt", padding=True).to(DEVICE)
            out = bert(**tokens)
            cls += out.last_hidden_state[0,0].cpu()
            cnt_cls += 1
            outputs[i:i+WINDOW] += out.last_hidden_state[0,1:-1].cpu()
            cnt_outputs[i:i+WINDOW] += 1
        if cnt_cls:
            cls = cls / cnt_cls
        outputs = outputs / cnt_outputs
        ans.append((cls, outputs))
    return ans

if __name__ == '__main__':
    a = getBERT(['你好。', '阿斯顿法国红酒看来。'])
    print(len(a))
    print(a[0][0].shape)
    print(a[0][1].shape)