import torch
import _global

from utils.load import Text
from utils.text2bert import getBERT

if __name__ == '__main__':

    text = Text()
    # print(text)
    ret, cls = getBERT(text)
    bert = {'ret':ret, 'cls':cls}
    # print(anhao)
    torch.save(bert, _global.bertPath)