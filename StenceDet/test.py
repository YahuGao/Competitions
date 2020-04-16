import torch
import torch.nn as nn
import torch.optim as optim
m = nn.Sequential(
      nn.Linear(3, 5),
      nn.ReLU(),
      nn.Linear(5, 3)
    )
weights, biases = [], []
for name, p in m.named_parameters():
    if 'bias' in name:
        biases += [p]
    else:
        weights += [p]

optim.SGD([
    {'params': weights},
    {'params': biases, 'weight_decay': 0}],
    lr=1e-2, momentum=0.9, weight_decay=1e-5)

torch.manual_seed(0)
print("parameters", m.parameters())
input = torch.randn(3, 5)
print("input", input)
target = torch.randn(3, 1)
print("target:", target)
optimizer = optim.SGD(m.parameters(), lr=0.1, weight_decay=0.5)
out = m(input)
print("output", out)
loss = torch.nn.MSELoss(out, target)
loss.backward()
optimizer.step()
print("parameters", m.parameters())

