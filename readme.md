# AIX2

寻找相关法律文书。第二版。

使用前需要在 `_global.py` 中修改对应的路径。

## 数据

### 需要的数据文件

使用前需要一些数据。

#### Raw data

- 原始的csv格式的数据库
- 从数据库中提取的，relWenshu非空的，可解析文书

“提取有relWenshu的文书” 的代码只用了一次，没有进行维护。

#### pre

生成这部分文件的代码只用了一次，没有维护。

`pre/content.pt`，从 Raw data 中提取的relWenshu非空的，文书的html的list。
`pre/labeledId.pt`，从 docID 到 content.pt下标 的 dict。
`pre/relation.pt`，content.pt中关联的文书对。

```python
import torch
a = torch.load('../project/datasets/pre/content.pt')
b = torch.load('../project/datasets/pre/labeledId.pt')
c = torch.load('../project/datasets/pre/relation.pt')
a[0] # (一个html字符串)
b['465937f130b4410493faa9d2009f441b'] # 0 ，docID对应的html是a[b[docID]]
c[0] # a[c[0][0]] 和 a[c[0][1]] 相关
```

#### all

划分的数据集。

`all/relation.pt`，每篇文书的编号，正例，负例。
`a;;/split.pt`，划分数据集。

```python
import torch
a = torch.load('../project/datasets/pre/content.pt')
b = torch.load('../project/datasets/all/relation.pt')
c = torch.load('../project/datasets/all/split.pt')
b[2]
# [10, [(305430, 2)], [(418974, 1), (58472, 3),...]]
# 这篇文书是10 (a[10])
# 有1个正样本305430 (a[305430])，匹配的关键字数量排名第2
# 有负样本418974 (a[418974])，匹配的关键字数量排名第1
c[0] # 0 第0文书需要被分在0(train set)中
c[5] # 1 第5文书需要被分在1(valid set)中
```
### 生成的数据文件

`python -m gen.html2text` 生成 `feature/text.pt`
`python -m text2anhao` 生成 `feature/anhao.pt`
`python -m text2bert` 生成 `feature/bert.pt`
`python -m text2entity` 生成 `feature/entity.pt`

## utils

在 utils 目录下。运行该目录下文件`A.py`主程序部分使用 `python -m utils.A`。

### raw2html.py

- `def readCSV(path)`: 传入csv的路径，返回列表html,列表info。

  - html是 csv文件中所有文书的html字符串 的list。
  - info是 对应文书的路径和docID 的list。

- `def magic(tmp)`：传入csv中CourtInfo字符串，返回      qwContent字符串。
  - 现在存在部分文书无法识别。
  - 运行逻辑为
    1. 判断是否乱码。
    2. 查找是否有`"qwContent":"`
    3. 将`"qwContent":"`到`","directory"`或`","viewCount"`中的`"`替换成`\"`，防止json解析失败。
    4. 读取json，返回`qwContent`的值。（含`DocInfoVo`的json没有处理）

### html2text.py

- `def html2text(html_text)`: 传入html字符串，返回一个字符串列表text，代表每句话的文本。
  - 使用BeautifulSoup解析html。
  - 手动删除 `<font>` 和 `<span>` 标签。
  - 删除空白符。
  - 忽略以 `['审判长', '审判员', '代理审判员', '人民陪审员', '书记员', '执行员']` 起始的行。
  - 使用换行 `\n` 连接句子。
  - 分段
    - 使用 `split_sentence_hard` 根据长度分段。每段话长为_global.textLength=160。
    - 若需要完整的文本，可以使用 `''.join(text)`。
- `def split_sentence(text)`: 传入文本，将其分句。
  - 可以调用 `split_sentence_hard` 或 `split_sentence_soft`。

### text2anhao.py

- `getAnHao(x)`: 传入文本字符串x, 返回提取的案号list。
  - 使用正则表达式提取案号。
  - 返回re.fineall格式化的结果。
  - example:
    - 输入 `[['张三在北京。', '张三在上海。\n', '阿巴阿巴。'], ]`。
    - 输出 `[[[('张三', 'PERSON', 0, 1), ('北京', 'LOCATION', 2, 3), ('张三', 'PERSON', 4, 5)], [('上海', 'LOCATION', 6, 7), ('阿巴阿巴', 'PERSON', 8, 9)], []]]`。
  

### text2bert.py

效果不好。

- `def getBERT(texts)`: 输入文本字符串列表texts，返回预训练模型的输出：列表ret，列表cls。
  - 调用 `BERT()`。
  - ret是每个位置的输出。
  - cls是[cls]位置的输出。
- `class BERT`: 加载和使用预训练模型。

### text2entity.py


- `def getEntity(texts)`: 输入一个列表texts，其中每个元素text是一个字符串列表，代表一篇文书中文本的每句话。输出列表key_words，其中每个元素是一个关键词列表。

### load.py

导入各种文件。
- `def Bert(path=_global.bertPath):`
- `def Html(path=_global.htmlPath):`
- `def Text(path=_global.textPath):`
- `def Anhao(path=_global.anhaoPath):`
- `def AllRelation(path=_global.allRelationPath):`
- `def AllSplit(path=_global.allSplitPath):`

### dataset.py

加载提取后的数据集。
```python
dataset = AllDataset()
train = AllSubset(dataset, 0)
valid = AllSubset(dataset, 1)
```
见 readme “数据”。
- `class AllDataset`:
- `class AllSubset(torch.utils.data.Dataset)`:

### examples.py

简易的生成例子的工具。

- `def example(l, name, num=20, same_anhao=False)`: 输入列表l，保存的目录名name，采样数n，选项same_anhao。
  - l中的元素形如 `(x,y)`。其中x,y是 *Html数据集* 中的文书编号。
  - 从l中随机选n个元素。
  - 将每个元素的html字符串保存
  - 若same_anhao 为 True，则还会保存两篇文书的相同的案号。

### test.py

用来测试模型的效果。

- `def predict(model, sample, th=0)`: 输入模型和模型的输入，以及threshold。输入预测的logits。

## gen

调用对应的utils中的代码，生成对应的文件。

- html2text.py
- text2anhao.py
- text2bert.py
- text2entity.py

## models

主要是classifier.py。一些模型。

## train

`train.py` 用于训练模型。

`python train.py --batch 16 --netType AttnBert`