import hanlp
import torch
import re

class NER:
    def __init__(self):
        HanLP = hanlp.load(hanlp.pretrained.mtl.OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
        tasks = list(HanLP.tasks.keys())
        # print(tasks)
        for task in tasks:
            if task not in ('tok', 'ner'):
                del HanLP[task]
        for task in HanLP.tasks.values():
            task.sampler_builder.batch_size = 64
        self.HanLP = HanLP
    def get(self, x):
        return [[b for b in a if b[1] in ('PERSON', 'LOCATION', 'ORGANIZATION')] for a in x]
    def __call__(self, x):
        # print(self.HanLP(x))
        # exit()
        return self.get(self.HanLP(x)['ner'])

def get_key(ner, plain_text_arr):
    if plain_text_arr == []:
        return []
    # starttime = time.perf_counter()

    # result = ner(plain_text_arr)
    # print(time.perf_counter() - starttime)
    # 699.0954339581076
    # return result

    sort_arr = list(enumerate(plain_text_arr))
    sort_arr.sort(key=lambda x: len(x[1]))
    sort_s = [x[1] for x in sort_arr]
    result = []
    # for i in tqdm(range(0, len(sort_s), 2048)):
    for i in range(0, len(sort_s), 2048):
        result.extend(ner(sort_s[i:i+2048]))
    result = [(x[0], x[1]) for x in zip(sort_arr, result)]
    result.sort(key=lambda x: x[0])
    # print(time.perf_counter() - starttime)
    # 673.7641270339955
    return [x[1] for x in result]

def getEntity(texts):
    a = []
    lines = []
    for text in texts:
        l = len(a)
        for i in range(len(text)):
            tmp = text[i]
            if i:
                tmp = text[i-1] + tmp
            if i != len(text) - 1:
                tmp = tmp + text[i+1]
            a.append(tmp)
        r = len(a)
        lines.append([l, r])
    # print(a)
    ner = NER()
    key_arr = get_key(ner, a)
    # print(key_arr)
    # return 
    key_words = []
    for (l, r), text in zip(lines, texts):
        now = key_arr[l:r]
        key_word = []
        for i in range(len(text)):
            start = 0
            end = len(text[i])
            if i:
                start += len(text[i-1])
                end += len(text[i-1])
            tmp = []
            for x in now[i]:
                if x[3] >= start and x[3] < end:
                    tmp.append(x)
            key_word.append(tmp)
        key_words.append(key_word)
    return key_words



if __name__ == '__main__':
    # a = getEntity([['张三在北京。', '张三在上海。\n', '阿巴阿巴。'], [' 阿巴阿巴是一个公司。']])
    import utils.load as load
    x = load.Text()[:10]
    a = getEntity(x)
    print(a)