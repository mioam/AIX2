import torch
import _global

from utils.load import Text
from utils.text2anhao import getAnHao

if __name__ == '__main__':

    text = Text()
    # print(text)
    anhao = [getAnHao(x) for x in text]
    # print(anhao)
    torch.save(anhao, _global.anhaoPath)