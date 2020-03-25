# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
import jieba


class Transformation:
    '''
    处理训练数据的类，某些情况下需要对训练的数据再一次的处理。
    如无需处理的话，不用实现该方法。
    '''

    def transformation_data(self, x_train=None, y_train=None, x_test=None, y_test=None):
        print("test.. Transformation")
        return x_train, y_train, x_test, y_test


class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance


class PreTrainedEmbedding(Singleton):
    embeddings = {}
    def __init__(self, preTrained_file='sgns.weibo.bigram-char'):

        with open(preTrained_file) as f:
            for line in tqdm(f):
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

        # print("%d words not found in %d words %.2f" % (count, len(words),
        #                                              count/len(words)))
        if len(words) < length:
            # print("padding %d vectors" % (length - len(words)))
            vector = self.embeddings.get('。')
            for i in range(length - len(words)):
                vectors.append(vector)

        vectors = np.array(vectors)
        return vectors


if __name__ == '__main__':
    embedding = PreTrainedEmbedding()
    exit()
