# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
from flyai.dataset import Dataset
from net import Net
import torch.nn.utils.rnn as rnn_utils

from path import MODEL_PATH

__import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net = None
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        if os.path.exists(self.net_path):
            self.net = torch.load(self.net_path)

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)

        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        x_data = rnn_utils.pad_sequence(x_data, batch_first=True)
        x_data = rnn_utils.pack_padded_sequence(x_data,
                                                [len(x_data[0])],
                                                batch_first=True)
        x_data = x_data.to(device)
        outputs = self.net(x_data)
        prediction = outputs.data.cpu().numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.net_path)
        labels = []
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        for data in datas:
            x_data = self.data.predict_data(**data)
            x_data = torch.from_numpy(x_data)
            if torch.numel(x_data) == 0:
                labels.append(0)
                continue
            x_data = rnn_utils.pad_sequence(x_data, batch_first=True)
            x_data = rnn_utils.pack_padded_sequence(x_data,
                                                    [len(x_data[0])],
                                                    batch_first=True)
            x_data = x_data.to(device)
            outputs = self.net(x_data)
            prediction = outputs.data.cpu().numpy()
            prediction = self.data.to_categorys(prediction)
            labels.append(prediction)
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))


if __name__ == '__main__':
    dataset = Dataset(10, 32)
    model = Model(dataset)
    x_vals, y_vals = dataset.evaluate_data_no_processor()
    labels = model.predict_all(x_vals)
    Y_vals = []
    for label in y_vals:
        if label['STANCE'] == 'NONE':
            Y_vals.append(0)
        elif label['STANCE'] == 'FAVOR':
            Y_vals.append(1)
        else:
            Y_vals.append(2)

    print(Y_vals)
    print(labels)
    from sklearn.metrics import f1_score
    print("f1_score: ", f1_score(Y_vals, labels, average='macro', labels=[0,1,2]))
