# -*- coding: utf-8 -*-
# @Time : 2019/5/30 下午5:53
# @Author : dabing
# @File : data_helper.py
# @Software: PyCharm

import os
import re
import numpy as np
import pandas as pd
import sys
import jieba


def data_clean(sent):
    """
    1. 数据清洗
    """
    if len(sent) == 0:
        print('[ERROR] data_clean faliled! | The params: {}'.format(sent))
        return None
    sentence = regular(sent)
    # sentence = remove_stop_words(sentence)
    return sentence

def regular(text):
    # 去除(# #)字段
    text = re.sub(r'#.*#',' ',text)
    # 去除多个@用户
    # unicode 编码中，中文范围为4e00-9fa5
    text = re.sub(r'@([\u4e00-\u9fa5a-zA-Z0-9_-]+)',' ',text)
    # 去除url
    text = re.sub(r'http[s]?://[a-zA-Z0-9.?/&=:]*', ' ', text)
    # 去除英文和数字
    # text = re.sub(r'[a-zA-Z0-9]', ' ', text)
    # 去除其他噪音字符
    text = re.sub(r'[—|（）()【】…「~_]+', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首行空格
    text = text.strip()

    return text

def remove_stop_words(text):
    new_text = []
    stop_words = []
    path = os.path.join(sys.path[0], 'stop_words.txt')
    with open(path,'r',encoding='utf-8') as file:
        for line in file.readlines():
            stop_words.append(line.strip())
    word_segment = jieba.lcut(text)

    for word in word_segment:
        if word not in stop_words:
            new_text.append(word)

    return ''.join(new_text)

''
if __name__ == "__main__":
    text = '#深圳禁摩限电# 自行车、汽车也同样会引发交通事故——为何单怪他们？（我早就发现：交通的混乱，反映了“交管局”内部的混乱！）必须先严整公安交管局内部！——抓问题的根本！@深圳交警@中国政府网@人民日报'
    text = data_clean(text)
    text = remove_stop_words(text)
    print(text)
    exit(0)
