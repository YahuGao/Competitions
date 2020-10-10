#!/usr/bin/env python

from parameters import config
from tools import load_data, data_generator
from net import Net, Evaluator

def do_train():
    train_data = load_data('./data/train.txt')
    val_data = load_data('./data/val.txt')

    net = Net()
    model = net.get_model()

    evaluator = Evaluator(val_data, config.model_save_to, model, net.CRF, net.NER)
    train_generator = data_generator(train_data, config.batch_size)

    history = model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=config.epochs,
        callbacks=[evaluator]
    )

    return history


if __name__ == '__main__':
    do_train()
