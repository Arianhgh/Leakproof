import torch

x = torch.randn(10, 3)
model = torch.nn.Linear(3, 1)
out = model(x)
