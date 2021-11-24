from os import pipe
from bs4 import BeautifulSoup
# import jionlp as jio
import hanlp

def html2text(html_text):
    # 预处理！
    soup = BeautifulSoup(html_text, 'html.parser')
    [s.unwrap() for s in soup('font')]
    [s.unwrap() for s in soup('span')]
    soup = BeautifulSoup(str(soup), 'html.parser')

    # text = soup.get_text()
    text = []
    for i in soup.stripped_strings:
        tmp = ''.join(i.split())
        if tmp.startswith(('审判长', '审判员', '代理审判员', '人民陪审员', '书记员', '执行员')):
            break
        tmp = hanlp.utils.rules.split_sentence(tmp)
        # tmp = jio.split_sentence(tmp, criterion='coarse')
        text.extend(tmp)
    # print(text)
    return text

if __name__ == '__main__':
    
    from utils.load import Html

    html = Html()[:5]
    text = [html2text(x) for x in html]
    # print(text)
