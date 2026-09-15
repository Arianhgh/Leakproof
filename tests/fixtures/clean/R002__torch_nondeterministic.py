import torch

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
x = torch.randn(10, 3)
model = torch.nn.Linear(3, 1)
out = model(x)
