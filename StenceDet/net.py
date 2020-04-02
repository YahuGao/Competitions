# -*- coding: utf-8 -*
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, input_size=300,
                 hidden_size=128,
                 num_layers=2,
                 drop_prob=0.5,
                 batch_first=True,
                 bidirectional=False,
                 output_size=3):
        super(Net, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers,
                            bidirectional=bidirectional,
                            batch_first=batch_first,
                            dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        out, _ = self.LSTM(input)
        out = self.fc(out)
        out = out[:, -1]
        return out

    def init_hidden(self, batch_size, device='cpu'):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size,
                             self.hidden_size).zero_().to(device),
                  weight.new(self.num_layers, batch_size,
                             self.hidden_size).zero_().to(device))
        return hidden


if __name__ == '__main__':
    input_size = 300
    hidden_size = 128
    num_layers = 6
    drop_prob = 0.5
    output_size = 3
    bidirectional = False
    batch_first = True

    if bidirectional:
        hidden_size *= 2
        num_layers *= 2

    batch_size = 5
    seq_len = 4
    inp = torch.randn(batch_size, seq_len, input_size)
    hidden_state = torch.randn(num_layers, batch_size, hidden_size)
    cell_state = torch.randn(num_layers, batch_size, hidden_size)
    hidden = (hidden_state, cell_state)

    model = Net(input_size, hidden_size,
                num_layers,
                drop_prob,
                batch_first,
                bidirectional,
                output_size)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    hidden = model.init_hidden(batch_size)
    model.to(device)
    lr = 0.0005
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out = model(inp)

'''
    # Test input and out shape
    out, (h_n, c_n) = model.LSTM(inp, hidden)
    print(out.size())
    print(h_n.size())
    print(c_n.size())
    out = model.dropout(out)
    print(out.size())
    out = model.fc(out)
    print(out.size())
'''
