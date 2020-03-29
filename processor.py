# -*- coding: utf-8 -*

import numpy as np
import torch
from flyai.processor.base import Base
import data_helper
import jieba
from flyai.utils import remote_helper

class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance


class PreTrainedEmbedding(Singleton):
    embeddings = {}
    def __init__(self, preTrained_file='sgns.weibo.bigram-char'):
        wordVecURL = 'https://www.flyai.com/m/sgns.weibo.word.bz2'
        path = remote_helper.get_remote_data(wordVecURL)
        with open('./data/input/model/sgns.weibo.bigram-char', 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                index = line.find(' ')
                word = line[:index]
                vector = np.array(line[index:].split(), dtype='float32')
                self.embeddings[word] = vector

    def turnToVectors(self, text, length=100):
        words = jieba.lcut(text)
        vectors = []
        count = 0
        for word in words:
            vector = self.embeddings.get(word, self.embeddings.get('的'))
            vectors.append(vector)
            if word not in self.embeddings.keys():
                count += 1

        if len(words) < length:
            vector = self.embeddings.get('。')
            for i in range(length - len(words)):
                vectors.append(vector)
        else:
            vectors = vectors[:100]

        vectors = np.array(vectors)
        if __name__ == '__main__':
            return vectors, count/len(words)
        return vectors


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
            return 0
        elif STANCE == 'FAVOR':
            return 1
        elif STANCE == 'AGAINST':
            return 2

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, TARGET, TEXT):
        text2vec = self.input_x(TARGET, TEXT)
        return text2vec

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        index = np.argmax(data)
        return index


if __name__ == '__main__':
    from flyai.dataset import Dataset
    dataset = Dataset(10, 32)
    train_x, train_y, val_x, val_y = dataset.get_all_data()
    preTrainedEmbedding = PreTrainedEmbedding()
    contents = [x['TEXT'] for x in train_x]
    unfounds = []
    for words in contents:
        print(words)
        vector, unfound = preTrainedEmbedding.turnToVectors(words)
        unfounds.append(unfound)
    print("unfound probability is: %f", np.mean(unfounds))
