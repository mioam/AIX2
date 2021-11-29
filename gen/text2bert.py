import torch
import _global

from utils.load import Text
from utils.text2bert import getBERT

if __name__ == '__main__':

    text = Text()
    # print(text)
    bert = getBERT(text)
    # print(anhao)
    torch.save(bert, _global.bertPath)