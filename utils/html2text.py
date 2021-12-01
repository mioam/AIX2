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
            continue
        text.append(tmp)
    text = '\n'.join(text)
    return split_sentence(text)

if __name__ == '__main__':
    
    from utils.load import Html

    html = ["<!DOCTYPE HTML PUBLIC -//W3C//DTD HTML 4.0 Transitional//EN'><HTML><HEAD><TITLE></TITLE></HEAD><BODY><div id='7'  style='TEXT-ALIGN: right; LINE-HEIGHT: 25pt; MARGIN: 0.5pt 36pt 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>书记员　　黄智锴</div><div style='TEXT-ALIGN: center; LINE-HEIGHT: 25pt; MARGIN: 0.5pt 0cm; FONT-FAMILY: 黑体; FONT-SIZE: 18pt;'>广东省广州市花都区人民法院</div><div style='TEXT-ALIGN: center; LINE-HEIGHT: 25pt; MARGIN: 0.5pt 0cm; FONT-FAMILY: 黑体; FONT-SIZE: 18pt;'>执 行 裁 定 书</div><div id='1'  style='TEXT-ALIGN: right; LINE-HEIGHT: 25pt; MARGIN: 0.5pt 0cm;  FONT-FAMILY: 宋体;FONT-SIZE: 15pt; '>（2018）粤0114执4076号之二</div><div id='2'  style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>申请执行人:广州市花都区狮岭大州无纺布商行,住所地广州市花都区狮岭镇合成村 自编合和路80号。</div><div style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>经营者：倪志溢。</div><div id='4'  style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>被执行人:邹清，女，1991年8月26日出生，汉族，住江西省吉安市新干县，</div><div id='4'  style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>被执行人:邓军，男，1987年3月8日出生，汉族，住江西省吉安市新干县，</div><div id='3'  style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>本院依据已经发生法律效力的（2017）粤0114民初4976号民事判决书，向被执行人发出执行通知书，责令被执行人履行义务。由于被执行人至今未能履行本院作出的民事判决书所确定的义务，根据《中华人民共和国民事诉讼法》第二百四十四条 、《最高人民法院关于适用<中华人民共和国民事诉讼法>的解释》第四百八十七条的规定，裁定如下：</div><div id='6'  style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>一、查封权属于被执行人邓军的号牌为粤Ｙ×××××的车辆档案。</div><div style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>二、查封期限为两年。</div><div id='4'  style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>需要续行查封的，应当在查封期限届满前十五日内向本院提出续行查封的书面申请；履行义务后可以申请解除查封。</div><div style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>本裁定送达后立即发生法律效力。</div><div style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>执行员王志文</div><div style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>二Ｏ一八年六月十二日</div><div id='7'  style='LINE-HEIGHT: 25pt; TEXT-INDENT: 30pt; MARGIN: 0.5pt 0cm;FONT-FAMILY: 宋体; FONT-SIZE: 15pt;'>书记员黄智锴</div></BODY></HTML>"]
    text = [html2text(x) for x in html]
    print(text)
