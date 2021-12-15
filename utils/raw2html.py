import csv
from tqdm import tqdm
import json

def magic(tmp):

    # # 删除乱码
    # if row[2].find('"s1":!') != -1:
    #     return None
    # if row[2].find('"s1":#') != -1:
    #     return None
    tmp = str(tmp)
    tmp = tmp[1:-1]
    tmp = tmp.replace('""','"')
    tmp = tmp.replace('\n','')
    tmp = tmp.replace('\\','\\\\')
    x = tmp.find('"qwContent":"')
    if x == -1: # 没有文本
        return None


    # 转义引号
    x += 13 # "qwContent":"
    y = tmp.find('","directory"',x)
    if y == -1:
        y = tmp.find('","viewCount"',x)
    tmp = tmp[:x] + tmp[x:y].replace(r'"',r'\"') + tmp[y:]

    # # 相关文书
    # # if tmp.find('"relWenshu"') == -1:
    # #     return None
    # # x0 = tmp.find('"relWenshu"')
    # # y0 = tmp.find(']',x0)+2
    # # if y0 - x0 != 15:
    # #     cnt += 1
    
    j = json.loads(tmp)
    if 'DocInfoVo' in j: # unknown format
        return None

    if j['qwContent'] == '':
        return None

    return j['qwContent']

def readCSV(path):
    errcnt = 0
    cnt = 0
    html = []
    info = []
    csv.field_size_limit(500 * 1024 * 1024)
    fcsv = csv.reader(open(path, 'r', encoding='utf-8'))
    flag = False
    for row in tqdm(fcsv):
        if not flag:
            flag = True
            continue
        try:
            qwContent = magic(row[1])
        except BaseException:
            errcnt += 1
            # print('ERROR', a, cnt)
            # with open('error', 'w', encoding='UTF-8') as f:
            #     f.write(row[1])
            
        if qwContent is not None:
            cnt += 1
            html.append(qwContent)
            info.append((path, row[0], row[1]))
    return html, info

if __name__ == '__main__':
    html, info = readCSV('../train_1.csv')
    from utils.html2text import html2text
    from utils.text2anhao import getAnHao
    text = [html2text(h) for h in html]
    anhao = [getAnHao(t) for t in text]
    # print(h)
    print(text[1])
    print(anhao[1])
    