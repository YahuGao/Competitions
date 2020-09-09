# -*- coding: utf-8 -*
import os
import sys

from flyai.processor.base import Base

import bert.tokenization as tokenization
# import jieba
from bert.run_classifier import convert_single_example_simple


class Processor(Base):
    def __init__(self):
        self.token = None

    def input_x(self, TARGET, TEXT):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        multi_cased_L-12_H-768_A-12 bert模型的一个种类
        chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
        chinese_L-12_H-768_A-12
        '''
#        print(STANCE)
        if self.token is None:
#            bert_vocab_file = os.path.join(sys.path[0],'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16', 'vocab.txt')
            bert_vocab_file = os.path.join(sys.path[0],'chinese_L-12_H-768_A-12','chinese_L-12_H-768_A-12', 'vocab.txt')
#            初始化了，只指定了bert_vocab_file
            self.token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
#            self.token = tokenization.FullTokenizer(vocab_file=bert_vocab_file)
#        TEXT = data_clean(TEXT)
            #max_seq_length=256
            

        return ""

    def input_y(self, STANCE):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        if STANCE == 'NONE':
            return 0
        elif STANCE == 'FAVOR':
            return 1
        elif STANCE == 'AGAINST':
            return 2

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return data[0]