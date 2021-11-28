import random
from utils.load import Anhao, Html

def example(l, name, num=20, same_anhao=False):
    html = Html()
    if same_anhao:
        anhao = Anhao()
    l = random.sample(l, num)
    for i, (x,y) in enumerate(l):
        # print(i)
        with open(f'{name}/{i}_x.html','w',encoding='utf8') as f:
            f.write(html[x])
        with open(f'{name}/{i}_y.html','w',encoding='utf8') as f:
            f.write(html[y])
        if same_anhao:
            with open(f'{name}/{i}_z.txt','w',encoding='utf8') as f:
                f.write(str(set(anhao[x]) & set(anhao[y])))