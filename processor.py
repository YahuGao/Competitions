# -*- coding: utf-8 -*

import numpy as np
from flyai.processor.base import Base
from transformation import PreTrainedEmbedding
import data_helper

'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):
    preTrainedEmbedding = PreTrainedEmbedding()
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_x(self, TARGET, TEXT):
        text = data_helper.data_clean(TEXT)
        text2vec = self.preTrainedEmbedding.turnToVectors(text)
        return text2vec

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, STANCE):
        if STANCE == 'NONE':
            return np.array([1, 0, 0])
        elif STANCE == 'FAVOR':
            return np.array([0, 1, 0])
        elif STANCE == 'AGAINST':
            return np.array([0, 0, 1])

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, TARGET, TEXT):
        return TARGET, TEXT

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        return np.argmax(data)
