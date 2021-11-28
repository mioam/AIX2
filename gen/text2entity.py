import torch
import _global

from utils.load import Text
from utils.text2entity import getEntity

if __name__ == '__main__':

    text = Text()[:4]
    # print(text)
    entity = getEntity(text)
    # print(anhao)
    torch.save(entity, _global.entityPath)