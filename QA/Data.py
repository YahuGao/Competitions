import json
import pandas as pd
import os
import jieba
import time

class Data:
    """
    对数据集进行处理，得到多种格式的数据：
    csv文件： 原问题和分词去停用词后的问题
    json文件： 将原问题保存成json格式，文件中是一个list
    list中的每一个元素为一个字典{title: 原问题， split: 切分后去停用词的问题}
    对于每个数据集， 输入数据集的名称和停用词，就能得到以上两个文件，方便复用
    """
    def __init__(self, filename, stoplist_name):
        self.cwd = os.getcwd()
        self.filename = filename
        self.stop_list = self.get_stoplist(stoplist_name)

    @staticmethod
    def get_stoplist(stoplist_name):
        """
        读取停用词
        @stoplist_name: 停用词的路径
        @return: list of stopwords
        """
        with open(stoplist_name, encoding='utf-8') as f:
            stop_word = f.readlines()
        stop_list = [x.strip() for x in stop_word]
        return stop_list

    def split_word(self, query):
        """
        @query: 带切分的问题
        @return: 
        """
        words = jieba.cut(query)
        result = ' '.join([word for word in words if word not in self.stop_list])
        return result

    def split_dataset(self):
        """
        对数据集所有的问题进行切词， title表示未切词的问题， split_title:表示切割后的问题
        @return: dataFrame of title and splited title
        """
        file_path = os.path.join(self.cwd, self.filename)
        with open(file_path, encoding='utf-8') as f:
            df_dataset = pd.read_csv(f)
        columns_name = list(df_dataset.columns)[0]
        df_dataset.rename(columns={columns_name: 'title'}, inplace=True)
        df_dataset['split_title'] = df_dataset['title'].apply(lambda x: self.split_word(x))
        save_filename = 'df_' + self.filename
        df_save_path = os.path.join(self.cwd, save_filename)
        df_dataset.to_csv(df_save_path, index=False)
        return df_dataset

    def excel2csv(self):
        # 获取用户上传数据的文件夹路径
        path = os.path.join(self.cwd, 'myuploads')
        # 默认路径下只有一个用户上传文件
        data_path = os.listdir(path)[0]
        if data_path.endswith('xlsx') or data_path.endswith('xls'):
            filepath = os.path.join(path, data_path)
            dataset = pd.read_excel(filepath)
            dataset.to_csv(os.path.join(self.cwd, 'qa.csv'), index=False)
        else:
            # 用户上传的不是excel文件时，提示
            pass

    def get_dataset(self):
        """
        读取json格式的数据
        """
        # 如果路径中存在了json数据就直接读取， 没有则对源数据进行处理
        json_path = os.path.join(self.cwd, self.filename[:-4] + '.json')
        save_filename = 'df_' + self.filename
        df_save_path = os.path.join(self.cwd, save_filename)
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
        else:
            if os.path.exists(df_save_path):
                df_dataset = pd.read_csv(df_save_path)
            else:
                df_dataset = self.split_dataset()
            
            data = df_dataset.to_dict(orient='record')

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)

def merge_json(insert_filename, total_filename = 'total_dataset.json'):
    """
    将新增的数据加入到json文件中
    @insert_filename: 保存新增数据的json文件
    @total_filename: 总数据的json文件
    """
    cwd = os.getcwd()
    total_path = os.path.join(cwd, total_filename)
    insert_path = os.path.join(cwd, insert_filename)
    with open(insert_path) as f:
        insert_data = json.load(f)

    if os.path.exists(total_path):
        with open(total_path) as f:
            total_data = json.load(f)
        total_data.extend(insert_data)
        with open(total_path, 'w') as f:
            json.dump(total_data, f)
    else:
        with open(total_path, 'w') as f:
            json.dump(insert_data, f)

if __name__ == '__main__':
    start = time.time()
    cwd = os.getcwd()
    print(cwd)
    filename = 'qa.csv'
    stoplist_name = 'stop_words.txt'
    data = Data(filename, stoplist_name)
    data.excel2csv()
    data.get_dataset()
    print('spent %.2f times to load data' % (time.time() - start))
    filename = 'train_questions.csv'
    data = Data(filename, stoplist_name)
    data.get_dataset()
