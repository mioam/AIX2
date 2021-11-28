from bs4 import BeautifulSoup
import hanlp
import _global
import re

MAX_LENGTH = _global.textLength

def split_sentence(text):
    return split_sentence_hard(text)

def split_sentence_hard(text):
    ret = []
    for i in range(0, len(text), MAX_LENGTH):
        ret.append(text[i:i+MAX_LENGTH])
    return ret

def split_sentence_soft(text):
    sentence = hanlp.utils.rules.split_sentence(text)
    ret = []
    for s in sentence:
        if len(s) > MAX_LENGTH:
            sr = re.sub('([,;，；])', r"\1\n", s)
            sr = sr.split('\n')
            l = MAX_LENGTH
            for x in sr:
                x = x.strip()
                if x == '':
                    continue
                if len(x) > MAX_LENGTH:
                    for t in range(0,len(x),MAX_LENGTH):
                        ret.append(x[t:t+MAX_LENGTH])
                    l = len(ret[-1])
                elif len(x) + l > MAX_LENGTH:
                    ret.append(x)
                    l = len(ret[-1])
                else:
                    ret[-1] = ret[-1] + x
                    l = len(ret[-1])
        else:
            ret.append(s)
    return ret


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
        text.append(tmp)
    text = '\n'.join(text)
    return split_sentence(text)

if __name__ == '__main__':
    
    from utils.load import Html

    html = Html()[:5]
    text = [html2text(x) for x in html]
    print(text)
