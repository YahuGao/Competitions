import keras
import bert4keras
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from parameters import config

def check_version():
    '''
    检查软件版本
    '''

    # sys.path.insert(0,'/notebook/.custom/TF2.1.0_JUPYTER2_gpu/pylib/Python3')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print("Keras version: %s", keras.__version__)
    print("Tensorflow version: %s", tf.__version__)
    print("bert4keras version %s", bert4keras.__version__)


def enable_gpu():
    '''
    指定使用GPU
    '''

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选用GPU序号
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def load_data(filename):
    '''
    从合并后的文件中读取数据
    '''
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()          # 读取整个文本内容
        for l in f.split('\n\n'):     # 将文本内容按行分隔
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    # 文本中字符和标签用空格分开， 获取当前行的字符和标签
                    char, this_flag = c.split(' ')
                except:
                    print(c)
                    continue
                # 当前标签和上一个标签都是O， 将当前行的字符添加到上一
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


class data_generator(DataGenerator):
    """
    数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [config.tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = config.tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < config.maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = config.label2id[l] * 2 + 1
                        I = config.label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [config.tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


if __name__ == '__main__':
    check_version()
