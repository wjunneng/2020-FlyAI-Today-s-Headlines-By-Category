# 2020-FlyAI-Today-s-Headlines-By-Category
2020 FlyAi 今日头条新闻分类

## BERTOrigin

'''
-e=2
-b=512
-score:88.35
'''

## BERTATT

'''
-e=2
-b=512
-score:88.45
'''

## BertCNNPlus

'''
-e=2
-b=512
-score:87.0
'''

## BertRCNN

'''
-e=2
-b=512
-score:87.35
'''

## BertATT + pseudo_labeling

'''
train:
-e=2
-b=512

predict:
-e=1
-b=512

-lr=5e-5
-score:88.85

------------

train:
-e=3
-b=512

predict:
-e=3
-b=512

-lr=2e-5
-score:88.90
'''

## BertATT + pseudo_labeling【虚标签】 + label_smoothing【标签平滑】 

'''
train:
-e=2
-b=512

predict:
-e=1
-b=512

-lr=5e-5
-score:88.80
'''
