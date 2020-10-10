import os
import codecs
import glob
from sklearn.model_selection import train_test_split


def _cut(sentence):
    """
    将一段文本根据标点切分成多个句子
    @sentence: 带切分的文本
    @return: 句子的列表
    """
    new_sentence = []      # 保存文本分割成的句子
    sen = []
    for i in sentence:
        # 先对文本按照句子结束符号进行分割
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))   # 每个句子以句号或者感叹号结束
            sen = []
            continue
        sen.append(i)

    # 若句子中没有句号，感叹号或者问号的， 使用逗号分割
    if len(new_sentence) <= 1:
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    # 将标点符号后面的内容加入句子列表
    if len(sen) > 0:
        new_sentence.append("".join(sen))
    return new_sentence


def cut_text_set(text_list, len_treshold=250):
    '''
    将文本按照标点符号切分成不超过最大长度的片段
    @text_list: 文本文件中的所有内容
    @len_treshold: 文本的长度阈值
    @return: 文本划分后的句子列表和句子的个数
    '''
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        # 若文件中文本的长度小于阈值， 直接添加到文本列表中， 否则切割处理文本
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)  # 一条数据被切分成多句话, 每句话不超过最大长度
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    # 文本划分后，若每句话长度不超过阈值，则连接到一起
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))  # 每段话被切割成的句子的个数
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


def from_ann2dic(r_ann_path, r_txt_path, w_path, w_file):
    '''
    给文本中的每个字符加上标签， 并按照'字符 标签'的格式写入新的文本文件
    字符和标签以空格分隔。
    @r_ann_path: 单个rnn文件路径
    @r_txt_path: 单个txt文件路径
    @w_path: 处理后存放数据的目录
    @w_file: 处理后存放数据的文件名前缀
    '''
    q_dic = {}
    with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n\r")
            # 实体ID号（以T开头）\t 实体类别 起始位置 结束位置\t实体内容
            # “实体ID号”和“实体类别”，“结束位置”和“实体内容”之间以tab分隔
            line_arr = line.split('\t')
            entityinfo = line_arr[1]      # 实体类别 起始位置 结束位置
            entityinfo = entityinfo.split(' ')  # “实体类别”、“起始位置”、“结束位置”以空格分隔
            cls = entityinfo[0]           # 实体类别
            start_index = int(entityinfo[1])  # 起始位置
            end_index = int(entityinfo[2])   # 结束位置
            length = end_index - start_index
            for r in range(length):
                if r == 0:
                    q_dic[start_index] = ("B-%s" % cls)  # 实体起始位置的标签为B-cls
                else:
                    q_dic[start_index + r] = ("I-%s" % cls)  # 实体其它位置标签为I-cls

    with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
        content_str = f.read()

    cut_text_list, _ = cut_text_set([content_str])

    i = 0  # 整个文本的字符计数器， 用于获得当前字符在整个文本中的位置
    for idx, line in enumerate(cut_text_list):
        w_path_ = "%s/%s-%s-new.txt" % (w_path, w_file, idx)
        with codecs.open(w_path_, "w", encoding="utf-8") as w:
            for str_ in line:
                # 跳过不可见字符
                if str_ in [" ", "", "\n", "\r"]:
                    pass
                else:
                    tag = q_dic.get(i, "O")
                    w.write('%s %s\n' % (str_, tag))
                i += 1
            w.write('%s\n' % "END O")


def process_dataset(files, write_path, data_dir):
    '''
    将文本和它的标注信息提取到单独的文件中存放，文本中的字符和标注处理成'字符 标签'的格式
    @files: a list of files in dataset
    @wirte_path: 处理后数据的存放目录
    @data_dir: 初始数据的存放目录
    '''
    for file in files:
        if file.find(".ann") == -1 and file.find(".txt") == -1:
            continue
        file_name = file.split('/')[-1].split('.')[0]
        read_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        read_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
        write_file = file_name
        from_ann2dic(read_ann_path, read_txt_path, write_path, write_file)


def merge_dataset(files_dir, files, write_path):
    '''
    将分散在各个文本中的'字符 标签'放到一个文件中
    @files_dir: 存放文件的目录
    @files: 存放'字符 标签'文本的列表
    @write_path: 合并后的文件存放路径（包括文件名）
    '''

    for fi in files:
        if not fi.endswith(".txt"):
            continue
        q_list = []
        print("开始读取文件:%s" % fi)
        fi = os.path.join(files_dir, fi)
        with codecs.open(fi, "r", encoding="utf-8") as f_read:
            line = f_read.readline()
            line = line.strip("\n\r")
            while line != "END O":
                q_list.append(line)
                line = f_read.readline()
                line = line.strip("\n\r")
        print("开始写入文本%s" % write_path)
        with codecs.open(write_path, "a", encoding="utf-8") as f_write:
            for item in q_list:
                if item.__contains__('\ufeff1'):
                    print("===============")
                f_write.write('%s\n' % item)
            f_write.write('\n')
        f_write.close()


if __name__ == '__main__':
    cwd = os.getcwd()      # 字符串后面没有 '/'
    data_dir = os.path.join(cwd, 'data')

    # 检查训练数据是否存在
    train_dir = os.path.join(cwd, 'data/train')
    if not os.path.exists(train_dir):
        print("DATA DIR NOT EXISTS!!!")
        exit(-1)

    print("split train and val dataset")
    # 读取训练数据中所有的文本，并划分训练文本和验证文本
    files = glob.glob(cwd + '/' + 'data/train/*.txt')
    train_files, val_files = train_test_split(
        files, test_size=0.2, random_state=2020)

    # 对训练数据进行处理，将各个文本中的字符和每个字符的标签写入单独的文本中
    processed_train_dir = os.path.join(
        data_dir, 'processed_train')  # 训练文本的保存目录
    if not os.path.exists(processed_train_dir):
        print("process train dataset")
        os.mkdir(processed_train_dir)
        # 处理训练数据
        process_dataset(train_files, processed_train_dir, train_dir)

    # 对验证数据进行处理，将各个文本中的字符和每个字符的标签写入单独的文本中
    processed_val_dir = os.path.join(data_dir, 'processed_val')  # 验证文本的保存目录
    if not os.path.exists(processed_val_dir):
        print("process val dataset")
        os.mkdir(processed_val_dir)
        # 处理验证数据
        process_dataset(val_files, processed_val_dir, train_dir)

    # 将处理后的所有训练文件内容合并到一个文件中
    merged_train_file = os.path.join(data_dir, 'train.txt')  # 合并后训练信息的保存文件
    if not os.path.exists(merged_train_file):
        print("merge train dataset")
        processed_train_files = os.listdir(processed_train_dir)  # 处理后的所有训练文件
        merge_dataset(processed_train_dir,
                      processed_train_files, merged_train_file)

    # 将处理后的所有验证文件内容合并到一个文件中
    merged_val_file = os.path.join(data_dir, 'val.txt')  # 合并后验证信息的保存文件
    if not os.path.exists(merged_val_file):
        print("merge val dataset")
        # 将处理后的所有验证文件内容合并到一个文件中
        processed_val_files = os.listdir(processed_val_dir)   # 处理后的所有验证文件
        merge_dataset(processed_val_dir, processed_val_files, merged_val_file)

    print("Done!")
