import gensim
import pandas as pd
import jieba
import os
import json
from Data import Data


class BaselineGensim(object):
    def __init__(self):
        self.cwd = os.getcwd()
        with open(os.path.join(self.cwd, 'train_questions.json')) as f:
            data = json.load(f)
            self.titles = []
            self.content = []
            for i in range(len(data)):
                self.titles.append(data[i]['title'])
                self.content.append(data[i]['split_title'])
        self.content_list = [str(line).rstrip().split() for line in self.content]
        with open(os.path.join(self.cwd, 'stop_words.txt'), encoding='utf-8') as f:
            lines = f.readlines()
        self.stop_list = [x.strip() for x in lines]
        # 加载词库字典， dic.token2id, dic.id2token
        self.dictionary = self.build_dictionary()
        # 根据字典中收集的特征， 将每个问题用BOW表示
        # 对每个不同单词的出现次数进行了计数，并将单词转换为其编号，然后以稀疏向量的形式返回结果。
        corpus = [self.dictionary.doc2bow(line) for line in self.content_list]
        num_features = max(self.dictionary.token2id.values())
        # 将每个问题由BOW表示转换为TFIDF表示
        self.tfidf = gensim.models.TfidfModel(corpus)
        self.idx = gensim.similarities.MatrixSimilarity(self.tfidf[corpus])

    @staticmethod
    def split_word(query, stop_list):
        words = jieba.cut(query)
        result = ' '.join([word for word in words if word not in stop_list])
        return result

    def build_dictionary(self):
        """
        得到一个基于数据集的字典， 字典中的key是每个词的编号, value是每个词
        """

        if os.path.exists(os.path.join(self.cwd, 'question_dictionary.dict')):
            dictionary = gensim.corpora.Dictionary.load(
                os.path.join(self.cwd, 'question_dictionary.dict'))
        else:
            content_list = [line.strip().split() for line in self.content]
            dictionary = gensim.corpora.Dictionary(content_list)
            dictionary.save(os.path.join(self.cwd, 'question_dictionary.dict'))
        return dictionary

    def get_topn_sims(self, sentences, n=5):
        """
        @sentences: 输入的问题
        @n: 找出n个最相似的问题
        @return: n个最相似的问题（未分词的）
        """
        split_sentence = self.split_word(sentences, self.stop_list).split()
        # 匹配结果1？？？？？
        # 对问题进行切词
        results_1 = {'title':sentences, 'split_title': split_sentence}
        # 根据字典转成BOW表示形式
        vec = self.dictionary.doc2bow(split_sentence)
        # 使用tfidf模型转成TFIDF表示形式
        # self.tfidf = gensim.models.TfidfModel(corpus)
        # 使用相似度矩阵计算句子和已有文本中的相似度
        # self.idx = gensim.similarities.MatrixSimilarity(self.tfidf[corpus])
        sims = self.idx[self.tfidf[vec]]
        # 把相似度转成元组的列表， 元组的元素是相似度和索引（索引对应着句子的编号）
        similarities = list(enumerate(sims))
        # 转成字典
        dict_sims = dict(similarities)
        # 对相似度进行逆序排序
        sorted_sims = sorted(dict_sims.values(), reverse=True)
        # 去除匹配到与原问题一样的结果，认为原问题的相似度一定最高，否则模型有问题
        top_sim_num = list(dict_sims.values()).index(sorted_sims[0])
        first = 0
        self.titles[top_sim_num] = ''.join(self.titles[top_sim_num].split())
        if sentences == self.titles[top_sim_num]:
            # 如果相似度最高的句子和原句子相等， 则从下一条开始匹配
            first += 1
        # 找出相似度最高的n个相似度
        topn_sims = sorted_sims[first:n+first]
        # 选出的n个问题的编号dict_sims:{index: similarities}
        topn_queries_num = [list(dict_sims.values()).index(i) for i in topn_sims]
        topn_queries = [self.titles[i] for i in topn_queries_num]
        topn_values = [dict_sims[i] for i in topn_queries_num]
        # 得到匹配结果2
        results_2 = []
        for i in range(n):
            each = [topn_queries_num[i], topn_queries[i], topn_values[i]]
            each_results_2 = {'index':str(topn_queries_num[i]), 'similarity':str(topn_values[i]),
                            'title':topn_queries[i]}
            results_2.append(each_results_2)
        # 汇总结果
        results = {'result1': results_1, 'result2':results_2}
        return results

if __name__ == '__main__':
    baseline = BaselineGensim()
    test_sentence = '国家税务总局关于调整中国国电集团公司合并纳税范围的通知'
    result = baseline.get_topn_sims(test_sentence)
    print(result)