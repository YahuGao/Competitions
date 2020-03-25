# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
from flyai.dataset import Dataset
from net import Net

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
        outputs = self.net(x_data)
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.net_path)
        datas = torch.Tensor(datas)
        outputs = self.net(datas)
        predictions = outputs.data.numpy()
        labels = []
        for prediction in predictions:
            label = self.data.to_categorys(prediction)
            labels.append(label)
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
    x_vals, y_vals = dataset.next_validation_batch()
    labels = model.predict_all(x_vals)
    print(labels)
