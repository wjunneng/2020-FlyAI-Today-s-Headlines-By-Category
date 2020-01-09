# -*- coding: utf-8 -*
"""
实现模型的调用
"""
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
# p = model.predict(news="赵丽颖很久没有登上微博热搜了，但你们别急，她只是在憋大招而已”！")
# 【缺了105/111】
# 106, 101, 116, 109, 108, 115
# 5, 1, 14, 8, 7, 13
#

p = model.predict_all(
    [{'news': '马云又出惊人言论，8年后房子最不值钱，他的话可信吗？'},
     {'news': '上联：泰山黄山赵本山，如何对下联？'},
     {'news': '为什么现在没有人玩CS了？'},
     {'news': '京东白条还款逾期了半个月，会影响个人征信吗？'},
     {'news': '长洲教育大家谈第二期主题公告, "社会主义,长洲教育大家谈,长洲,核心价值观,核心价值观融入"'},
     {'news': '脐橙卷叶？原因竟然是这个！, "卷叶,脐橙,线虫,脐橙卷叶,脐橙卷"'}])

print(p)
