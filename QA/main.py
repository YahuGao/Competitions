#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yahu Gao

from BaselineGensim import BaselineGensim
# from Wmd_model import Wmd_model
from Filter_sentence import load_target_sentence, get_same_rate, length_difference_rate, compute_inverse
import os, json

def interface(input_sentence, model_name, n=5) -> object:
    cwd = os.getcwd()
    with open(os.path.join(cwd, 'train_questions.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
    if model_name == 'tfidf':
        model = BaselineGensim()
        results = model.get_topn_sims(input_sentence, n)
    else:
        print('不存在这个模型')
    '''
    elif model_name == 'wmd':
        model = Wmd_model(data)
        results = model.GetSimilarity(input_sentence, n)
    '''

    
    target_sentence, target_key = load_target_sentence(results, n)
    confidence = True   # 置信度
    if target_sentence != 0:            # 最高相似度小于阈值
        len_diff_rate = length_difference_rate(input_sentence, target_sentence)
        same_rate = get_same_rate(input_sentence, target_sentence, model.stop_list)
        inverse_rate = compute_inverse(input_sentence, target_sentence, model.stop_list)
        inconfidence_num = 0
        if len_diff_rate > 2:   # 两个问题长度相差100， 可调整
            inconfidence_num += 1
        if same_rate < 0.5:
            inconfidence_num += 1
        if inverse_rate > 0.5:
            inconfidence_num += 1
        if inconfidence_num >=2:    # 如果有两个以上的检查认为目标句和query不相似
            confidence = False
        # 找到对应的要筛选的句子,形如{'index':编号,'similarity':相似度，'title':问题}
        target_result_2 = results['result2'][target_key]
        if not confidence:
            target_result_2['confidence'] = '不准确'
        else:
            target_result_2['confidence'] = '准确'

    results_string = json.dumps(results, ensure_ascii=False)
    return results_string

if __name__ == "__main__":
    test_sentence = u'“三证合一、一照一码”登记制度改革是什么？具体推行时间和范围是如何规定的'
    sent = u'“三证合一、一照一码”登记制度' 
    sentences = u'财政部国家税务总局关于全面推开营业税改征增值税试点的通知'
    model_name = 'tfidf'
    n = 5
    results = interface(sentences, model_name, n)
    print(results)
