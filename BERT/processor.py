# -*- coding: utf-8 -*
from flyai.processor.base import Base


class Processor(Base):
    def __init__(self):
        self.id2lael = ['news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house', 'news_car',
                        'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world', 'news_agriculture',
                        'news_game', 'stock', 'news_story']

    def input_x(self, news):
        """
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        """
        return news

    def input_y(self, category):
        """
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        """
        label_hot = [0 for i in range(15)]
        label_hot[self.id2lael.index(category.strip('\n'))] = 1
        return label_hot

    def output_y(self, index):
        """
        验证时使用，把模型输出的y转为对应的结果
        """
        return self.id2lael[int(index)]
