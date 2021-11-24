import re

def getAnHao(x):
    if isinstance(x,str):
        x = [x]
    ret = []
    
    收案年度 = r'[0-9]+'
    法院代字 = r'最高法|[内军兵京津沪渝藏宁新桂黑吉辽晋冀陕甘川黔云琼浙鲁苏皖闽赣豫鄂湘粤青港澳台][0-9\*]*'
    类型代字 = r'[\u4E00-\u9FFF]{1,3}'
    案件编号 = r'[0-9０１２３４５６７８９、\-]+'
    pattern = f'[\(（〔]?({收案年度})[\)）〕]?({法院代字})({类型代字})({案件编号})号'
    # pattern = r'[\(（]'+ 收案年度 + r'[\)）]' + 法院代字 + 类型代字 + 案件编号 + '号'
    # print(pattern)
    for a in x:
        ret.extend(tuple(re.findall(pattern, a)))
        # TODO
    return ret

if __name__ ==  '__main__':
    import utils.load as load
    x = load.Content()[:10]
    print(getAnHao(x))