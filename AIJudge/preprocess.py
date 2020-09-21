import jieba
import jieba.analyse
import jieba.posseg
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

def get_stopwords(path='./data/stop_words.txt'):
    '''
    @path: 停用词文件路径
    @return: 停用词列表
    '''
    stopwords = []
    with open(path) as f:
        for word in f.readlines():
            stopwords.append(word.strip())

    return stopwords


def split_word(text, stopwords=get_stopwords()):
    '''
    @text: 待分词的字符串
    @stopwords: 停用词列表
    @return: 分词，去停用词后空格相连的句子
    '''
    result = ' '.join([word.strip() for word in jieba.cut(text)
                       if word.strip() not in stopwords])
    return result


def load_data(train='./data/train.csv', test='./data/test.csv', nrows=None):
    df_tr = pd.read_csv(train, sep='\t', nrows=nrows).fillna(0)
    df_te = pd.read_csv(test, sep='\t', nrows=nrows).fillna(0)

    df_tr['text'] = df_tr['text'].parallel_apply(split_word)
    df_te['text'] = df_te['text'].parallel_apply(split_word)
    return df_tr, df_te


if __name__ == '__main__':
    df_tr, df_te = load_data(nrows=100)
    print(df_tr[:2])
    print(df_tr.columns)
    print(df_te[:2])
    print(df_te.columns)
