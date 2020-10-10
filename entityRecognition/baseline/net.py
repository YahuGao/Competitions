from parameters import config
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense, Bidirectional, LSTM
from keras.models import Model
from tqdm import tqdm
import numpy as np
from bert4keras.backend import keras, K


class NamedEntityRecognizer(ViterbiDecoder):
    """
    命名实体识别器
    """
    def recognize(self, text, model):
        tokens = config.tokenizer.tokenize(text)
        mapping = config.tokenizer.rematch(text, tokens)
        token_ids = config.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], config.id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


class Net():
    '''
    定义模型的网络结构
    '''

    def __init__(self):
        self.CRF = ConditionalRandomField(lr_multiplier=config.crf_lr_multiplier)
        self.model = self.get_model()
        self.NER = NamedEntityRecognizer(trans=K.eval(self.CRF.trans), starts=[0], ends=[0])


    def get_model(self):
        pretrained_bert = build_transformer_model(
                                config.bert_config_path,
                                config.bert_checkpoint_path,
                            )

        pretrained_bert.trainable = True
        set_trainable = False
        for layer in pretrained_bert.layers:
            if (layer.name.startswith('Transformer-10') or
                layer.name.startswith('Transformer-11') or
                layer.name.startswith('Transformer-9')):
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        last_layer1 = 'Transformer-%s-FeedForward-Norm' % (config.bert_layers - 1)
        output_layer1 = pretrained_bert.get_layer(last_layer1).output
        last_layer2 = 'Transformer-%s-FeedForward-Norm' % (config.bert_layers - 2)
        output_layer2 = pretrained_bert.get_layer(last_layer2).output
        last_layer3 = 'Transformer-%s-FeedForward-Norm' % (config.bert_layers - 3)
        output_layer3 = pretrained_bert.get_layer(last_layer3).output
        output = keras.layers.add([output_layer1, output_layer2, output_layer3])

        output = Bidirectional(LSTM(128, return_sequences=True))(output)
        output = Dense(config.num_labels)(output) # 27分类

        output = self.CRF(output)

        model = Model(pretrained_bert.input, output)

        model.compile(
            loss=self.CRF.sparse_loss,
            optimizer=Adam(config.learning_rate),
            metrics=[self.CRF.sparse_accuracy]
        )

        return model


def evaluate(data, NER, model):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text, model)) # 预测
        T = set([tuple(i) for i in d if i[1] != 'O']) #真实
        X += len(R & T) 
        Y += len(R) 
        Z += len(T)
    precision, recall =  X / Y, X / Z
    f1 = 2*precision*recall/(precision+recall)
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self,valid_data, model_save_to, model, CRF, NER):
        self.best_val_f1 = 0
        self.valid_data = valid_data
        self.scheduler_patience = 1
        self.early_stop_patience = 3
        self.model_save_to = model_save_to
        self.pre_f1 = 0 # 保存上一轮的f1
        self.bad_count = 3 # 记录f1比上一轮差的次数， 达到3次后重置学习率，试图跳出局部最优
        self.model = model
        self.CRF = CRF
        self.NER = NER

    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        trans = K.eval(self.CRF.trans)
        self.NER.trans = trans
        # print(NER.trans)
        f1, precision, recall = evaluate(self.valid_data, self.NER, self.model)
        if f1 >= self.best_val_f1:
            # 取得新的最优f1, 更新最优f1, 保存模型
            self.best_val_f1 = f1
            self.model.save_weights(self.model_save_to)
        print(
            'F1: %.5f, P: %.5f, R: %.5f, best f1: %.5f, lr: %.6f\n' %
            (f1, precision, recall, self.best_val_f1, lr))
        
        if True:   # 学习率调整策略0（学习率逐步降低0.3,当学习率接近0时，固定学习率）
            if lr * 0.7 > 0.000001:
                K.set_value(self.model.optimizer.lr, lr * 0.7)
        if False:   # 学习率调整策略1（学习率逐步降低0.3,当学习率接近0时，重置学习率）
            if lr * 0.7 >= 0.000001:
                K.set_value(self.model.optimizer.lr, lr * 0.7)
            else:
                K.set_value(self.model.optimizer.lr, 1e-4)
        if False:  # 学习率调整策略2（若f1降低， 则降低学习率， 当学习率接近0时，重置学习率）
            if f1 >= self.pre_f1:
                # 若f1 优于上一轮，重置早停, bad_count和patience计数器
                self.scheduler_patience = 1
                self.early_stop_patience = 3
                self.bad_count = 3
            else:
                self.scheduler_patience -= 1
                self.early_stop_patience -= 1
                self.bad_count -= 1
                if self.early_stop_patience == 0:
                    pass # 去除早停
                    exit()
                # 若f1比上一轮的结果差，则降低学习率
                if self.scheduler_patience == 0:
                    #　若学习率过低， 则重置学习率
                    if lr * 0.7 >= 0.000001:
                        K.set_value(self.model.optimizer.lr, lr * 0.7)
                    else:
                        K.set_value(self.model.optimizer.lr, 1e-4)
                    # 调整后，重置patience计数器
                    self.scheduler_patience = 1
                if self.bad_count == 0:
                    K.set_value(self.model.optimizer.lr, 1e-4)
                    # 调整后，重置bad_count计数器
                    self.bad_count = 3

            # 更新pre_f1
            self.pre_f1 = f1
