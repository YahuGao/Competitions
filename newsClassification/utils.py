import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def check_device():
    # 查看有效的CPU和GPU
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
    print(device_lib.list_local_devices())


def assign_gpu():
    # 指定使用GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选用GPU序号
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def create_f1():
    def f1_function(y_true, y_pred):
        y_pred_binary = tf.where(y_pred >= 0.5, 1., 0.)
        # Avoid TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'.
        tp = tf.reduce_sum(y_true * y_pred_binary)
        y_true = tf.cast(y_true, dtype=tf.float32)
        predicted_positives = tf.reduce_sum(y_pred_binary)
        possible_positives = tf.reduce_sum(y_true)
        return tp, predicted_positives, possible_positives
    return f1_function


class F1_score(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # handles base args (e.g., dtype)
        self.f1_function = create_f1()
        self.tp_count = self.add_weight("tp_count", initializer="zeros")
        self.all_predicted_positives = self.add_weight(
            'all_predicted_positives', initializer='zeros')
        self.all_possible_positives = self.add_weight(
            'all_possible_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, predicted_positives, possible_positives = self.f1_function(
            y_true, y_pred)
        self.tp_count.assign_add(tp)
        self.all_predicted_positives.assign_add(predicted_positives)
        self.all_possible_positives.assign_add(possible_positives)

    def result(self):
        precision = self.tp_count / self.all_predicted_positives
        recall = self.tp_count / self.all_possible_positives
        f1 = 2*(precision*recall)/(precision+recall)
        return f1


# 定义keras的附带权重信息的数据生成器
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y=None, n_classes=None, maxlen=400, batch_size=32):
        '''Initialization'''
        self.x = x
        self.y = y
        self.n_classes = n_classes
        self.maxlen = maxlen
        self.batch_size = batch_size
        if self.y:
            self.sample_weights = compute_sample_weight("balanced", self.y)

    def __len__(self):
        ''' Denotes the number of batches per epoch'''
        # 必须进行整型转换
        return int(np.floor(len(self.x) / self.batch_size))

    def __data_process(self, inputs):
        '''Generate data containing batch_size samples'''
        X = []
        for x in inputs:
            # 将字符串转成整数的列表
            sentence = [int(num)+1 for num in x.split()]
            X.append(sentence)
        return X

    # 一个batch的数据处理，返回需要feed到模型中训练的数据
    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = range(index*self.batch_size, (index+1)*self.batch_size)

        # Get inputs and labels from original data
        x = [self.x[index] for index in indexes]

        # Process inputs if needed
        x = self.__data_process(x)

        # padding
        x = sequence.pad_sequences(x, self.maxlen)

        # Transfer type to numpy.ndarray
        x = np.array(x)
        # for HAN
        x.reshape((self.batch_size, 1, self.maxlen))

        if self.y:
            y = [self.y[index] for index in indexes]
            sample_weights = [self.sample_weights[index] for index in indexes]
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
            sample_weights = np.array(sample_weights)

            return x, y, sample_weights
        return x

# 定义keras的附带权重信息的数据生成器, 针对HAN网络


class DataGeneratorHAN(keras.utils.Sequence):
    def __init__(self, x, y=None, n_classes=None, batch_size=32, maxlen_text=16, maxlen_sentence=25):
        '''Initialization'''
        self.x = x
        self.y = y
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.maxlen_text = maxlen_text
        self.maxlen_sentence = maxlen_sentence
        if self.y:
            self.sample_weights = compute_sample_weight("balanced", self.y)

    def __len__(self):
        ''' Denotes the number of batches per epoch'''
        # 必须进行整型转换
        return int(np.floor(len(self.x) / self.batch_size))

    def __data_process(self, inputs):
        '''Generate data containing batch_size samples'''
        X = []
        for x in inputs:
            # 将字符串转成整数的列表
            sentence = [int(num)+1 for num in x.split()]
            sentence = np.array(sentence)
            X.append(sentence)

        return X

    # 一个batch的数据处理，返回需要feed到模型中训练的数据
    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = range(index*self.batch_size, (index+1)*self.batch_size)

        # Get inputs and labels from original data
        x = [self.x[index] for index in indexes]

        # Process inputs if needed
        x = self.__data_process(x)

        # padding
        x = sequence.pad_sequences(
            x, self.maxlen_text * self.maxlen_sentence)

        # Transfer type to numpy.ndarray
        x = np.array(x)
        # for HAN
        x = x.reshape((len(x), self.maxlen_text, self.maxlen_sentence))

        if self.y:
            y = [self.y[index] for index in indexes]
            sample_weights = [self.sample_weights[index] for index in indexes]
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
            sample_weights = np.array(sample_weights)

            return x, y, sample_weights
        return x
