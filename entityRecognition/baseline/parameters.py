import os
from datetime import datetime
from bert4keras.tokenizers import Tokenizer

def get_model_name(maxlen, epochs, batch_size, bert_layers):
    '''
    根据文本最大长度、epochs、batch_size、bert层数和训练开始时间确定模型的名字
    @return: 模型的名字
    '''

    date_time = str(datetime.now())
    start_time = date_time.replace(' ', '_').replace(':', '_')[:16]
    super_parameters = (str(maxlen) + "_"
                        + str(epochs) + "_"
                        + str(batch_size) + "_"
                        + str(bert_layers))
    model_name = super_parameters + "_" + start_time + ".weights"
    return model_name


class SuperParameters():
    def __init__(self,
                 maxlen=500,
                 epochs=10,
                 batch_size=16,
                 bert_layers=12,
                 learning_rate=1e-4,  # bert_layers越小，学习率应该要越大
                 crf_lr_multiplier=1000,  # 必要时扩大CRF层的学习率
                 bert_config_path = '/data/bert/bert_config.json',
                 bert_checkpoint_path = '/data/bert/bert_model.ckpt',
                 bert_dict_path = '/data/bert/vocab.txt',
                ):
        self.maxlen = maxlen
        self.epochs = epochs
        self.batch_size = batch_size
        self.bert_layers = bert_layers
        self.learning_rate = learning_rate
        self.crf_lr_multiplier = crf_lr_multiplier
        self.bert_config_path = bert_config_path
        self.bert_checkpoint_path = bert_checkpoint_path
        self.bert_dict_path = bert_dict_path

        self.model_name =  get_model_name(maxlen, epochs, batch_size, bert_layers)
        cwd = os.getcwd()
        weights_path = os.path.join(cwd, 'weights')
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        self.model_save_to = os.path.join(weights_path, self.model_name)
        print("Best model will saved as %s" % self.model_save_to)

        self.labels = ['SYMPTOM',
                        'DRUG_EFFICACY',
                        'PERSON_GROUP',
                        'SYNDROME',
                        'DRUG_TASTE',
                        'DISEASE',
                        'DRUG_DOSAGE',
                        'DRUG_INGREDIENT',
                        'FOOD_GROUP',
                        'DISEASE_GROUP',
                        'DRUG',
                        'FOOD',
                        'DRUG_GROUP']

        self.id2label = dict(enumerate(self.labels))
        self.label2id = {j: i for i, j in self.id2label.items()}
        self.num_labels = len(self.labels) * 2 + 1

        self.tokenizer = Tokenizer(self.bert_dict_path, do_lower_case=True)


config = SuperParameters()
