import os

device = 'cpu'
datasetsPath = '../project/datasets'

htmlPath = os.path.join(datasetsPath,'pre','content.pt')

bertPath = os.path.join(datasetsPath,'feature','bert.pt')
textPath = os.path.join(datasetsPath,'feature','text.pt')
anhaoPath = os.path.join(datasetsPath,'feature','anhao.pt')

allRelationPath = os.path.join(datasetsPath,'all','relation.pt')
allSplitPath = os.path.join(datasetsPath,'all','split.pt')