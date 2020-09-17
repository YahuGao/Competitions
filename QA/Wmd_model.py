#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from time import time
import pickle
from gensim.models import Word2Vec
import jieba
from gensim.similarities import WmdSimilarity

import os
import json


class Wmd_model:
    def __init__(self, data, w2v_size=64):
        self.cwd = os.getcwd()
        with open(os.path.join(self.cwd, 'stop_words.txt'), encoding='utf-8') as f:
            lines = f.readlines()
        self.stop_list = [x.strip() for x in lines]

        self.data = data    # json格式的数据集
        self.titles = []    # 未分词的问题
        self.content = []   # 已分词去停用词的问题
        for i in range(len(self.data)):
            if isinstance(self.data[i]["split_title"], str):
                self.titles.append(self.data[i]["title"])
                self.content.append(self.data[i]["split_title"])

        # 使用content数据训练词向量word2vec, w2v_size 为词向量大小
        w2v_start = time()
        self.w2v_model = Word2Vec(self.content, workers=2, size=w2v_size)
        w2v_end = time()
        print('w2v took %.2f seconds to run.' % (w2v_end - w2v_start))

    def split_word(self, query):
        """
        结巴分词， 去停用词
        @query: 待切问题
        @stop_list: 停用词表
        return: 空格连接的切分后去停用词的字符串
        """
        words = jieba.cut(query)
        result = ' '.join([word for word in words if word not in self.stop_list])
        return result

    def get_similarity(self, sentences, num_best=5):
        """
        @sentences: 用户输入的问题
        @num_best: 要获取的相似问题的个数
        """
        self.num_best = num_best
        start = time()
        # 初始化WmdSimilarity对象， 匹配num_best+1条句子， 如果匹配到原问题就去掉
        similaritier = WmdSimilarity(self.content, self.w2v_model, num_best=self.num_best+1)
        split_sent = self.split_word(sentences)
        results_1 = {'title': sentences, 'split_title': split_sent.split()}
        sims = similaritier[split_sent]

        # 相似度最高的放在第0个
        top_sim_num = sims[0][0]    # 相似度最高的问题的编号
        # 相似度最高的问题
        self.titles[top_sim_num] = ' '.join(self.titles[top_sim_num].split())
        # 如果相似度最高的问题和用户输入的问题一致， 则删除， 否则去除最后一条
        if sentences == self.titles[top_sim_num]:
            sims.remove(sims[0])
        else:
            sims = sims[:-1]
        
        results_2 = []
        for i, sim in enumerate(sims):
            questions_num = sim[0]      # 匹配问题的编号
            question = self.titles[questions_num]    # 根据编号找到问题
            each_result_2 = {'index': str(questions_num), 'similarity': str(sim[1]),
                            'title': question, 'confidence': None}
            results_2.append(each_result_2)

        # 汇总结果1 和结果2
        results = {'result1': results_1, 'result2': results_2}
        print('Cell took %.2f seconds to run.' % (time() - start))
        return results


if __name__ == '__main__':
    cwd = os.getcwd()
    with open(os.path.join(cwd, 'train_questions.json'), 'r') as f:
        data = json.load(f)
    wmd_model = Wmd_model(data)
    sent = '“三证合一、一照一码”登记制度改革是什么？具体推行时间和范围是如何规定的' 
    results = wmd_model.get_similarity(sent)
    print(results)