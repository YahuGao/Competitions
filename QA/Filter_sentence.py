import pandas as pd
import jieba

'''
对得到的结果进行删选
'''

def load_target_sentence(results, n=5):
    '''
    results: {'result1':result1,
                'result2':[{'index': index, 'similarity':similarity, 'title':title}]}
    从模型返回的n个与问题中找出相似度最高且大于特定阈值（0.8）的问题
    若最大相似度小于阈值则返回0 和最高相似度的索引
    @results: 从模型返回的n个与问题最相似的问题
    '''
    target = -1
    target_key = -1
    # result2: list of dict with {'index': index, 'similarity':similarity, 'title':title}
    result2 = results['result2']
    max_sim_index = 0
    max_sim = 0
    for i in range(n):
        each_result_2 = result2[i]
        if float(each_result_2['similarity']) > float(max_sim):
            max_sim = each_result_2['similarity']
            max_sim_index = i
    if float(max_sim) > 0.8:        # 只有当问题相似度大于某一阈值时，确定找到目标
        target = result2[max_sim_index]
        target_key = max_sim_index
    if target != -1:
        target_sentence = target['title']
    else:
        target_sentence = 0
    return target_sentence, target_key

def get_same_rate(input_sentence, target_sentence, stop_list):
    """
    @input_sentence: 用户输入问题
    @target_sentence: 要过滤的问题
    return： 过滤问题和用户输入问题的相同率
    """
    input_words = []
    target_words = []
    input_length = len(input_sentence)
    for i in input_sentence:
        if i not in stop_list:
            input_words.append(i)
    for i in target_words:
        if i not in stop_list:
            target_words.append(i)
    same_words = 0
    for i in target_words:
        if i in input_words:
            same_words += 1
    same_words_rate = same_words / input_length

    return same_words_rate

def length_difference_rate(input_sentence, target_sentence):
    input_len = len(input_sentence)
    target_len = len(target_sentence)
    diff = abs(input_len - target_len)
    len_diff_rate = diff / input_len
    return len_diff_rate

def compute_inverse(input_sentence, target_sentence, stop_list):
    """
    @input_sentence: 用户输入问题
    @target_sentence: 要过滤的问题
    return： 过滤问题和用户输入问题的逆序数
    """
    # 切词去停用词
    input_list = [word for word in jieba.cut(input_sentence) if word not in stop_list]
    target_list = [word for word in jieba.cut(target_sentence) if word not in stop_list]
    # 将输入句子的没歌词从小到大编号
    keys = list(range(len(input_list)))
    # 生成索引和输入句子中的词组成的字典
    input_dict = {index: word for index, word in enumerate(input_list)}
    # 生成目标句子的编号序列， 如果目标句子中的词在输入句子中， 则该词的编号是输入句子中的该词的编号，如果不在则忽略
    target_index = []
    values_list = list(input_dict.values())
    for i in target_list:
        if i in values_list:
            i_index = values_list.index(i)
            target_index.append(i_index)
    inverse_number = 0
    for i, index in enumerate(target_index):
        for forward_index in target_index[i + 1:]:
            if forward_index < index:
                inverse_number += 1
    inverse_rate = inverse_number / len(target_index)
    return inverse_rate
