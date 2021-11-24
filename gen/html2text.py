import torch
from tqdm import tqdm
import _global

from utils.load import Html
from utils.html2text import html2text

if __name__ == '__main__':
    html = Html()
    # text = [html2text(x) for x in tqdm(html)]

    from multiprocessing import Pool

    with Pool(3) as pool:
        text = pool.map(html2text, html)

    torch.save(text, _global.textPath)