# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: Yahu
"""

import argparse
import torch
from flyai.dataset import Dataset
from model import Model
from net import Net
import numpy as np
from path import MODEL_PATH
from data_helper import FlyAIDataSet

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

'''
实现自己的网络机构
'''
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

input_size = 300
hidden_size = 128
num_layers = 2
drop_prob = 0.5
output_size = 3
bidirectional = False
batch_first = True

if bidirectional:
    hidden_size *= 2
    num_layers *= 2

net = Net(input_size,
          hidden_size,
          num_layers,
          drop_prob,
          batch_first,
          bidirectional,
          output_size).to(device)

lr = 0.0005
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
clip = 5
counter = 0
print_every = 10
valid_loss_min = np.Inf

train_x, train_y, val_x, val_y = dataset.get_all_processor_data()
train_dataset = FlyAIDataSet(train_x, train_y)
val_dataset = FlyAIDataSet(val_x, val_y)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                           batch_size=args.BATCH)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True,
                                         batch_size=args.BATCH)

'''
dataset.get_step() 获取数据的总迭代次数
实现自己的模型保存逻辑
'''

for i in range(args.EPOCHS):
    best_score = 0
    h = net.init_hidden(args.BATCH, device)
    for inputs, labels in train_loader:
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        net.zero_grad()
        out = net(inputs)
        loss = criterion(out, torch.max(labels, 1)[1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            val_h = net.init_hidden(args.BATCH, device)
            val_losses = []
            net.eval()
            for inp, lab in val_loader:
                inp, lab = inp.to(device), lab.to(device)
                out = net(inp)
                val_loss = criterion(out, torch.max(lab, 1)[1])
                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(i+1, args.EPOCHS),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                model.save_model(net, MODEL_PATH, overwrite=True)
                print('Validation loss decreased({: .6f} --> {: .6f}). \
                      Saving model ...'.format(
                      valid_loss_min, np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

model.save_model(net, MODEL_PATH, overwrite=False)
