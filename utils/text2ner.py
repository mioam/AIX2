import hanlp
import torch

#  TODO
def getNER(texts):
    ans = []
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    # out = HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
    # print(out)
    # exit()
    for text in texts:
        result = HanLP(text)['ner/msra']
        # print(result)
        ans.append(result)

    return ans

if __name__ == '__main__':
    a = getNER(['张三在北京。'])
    print(a)