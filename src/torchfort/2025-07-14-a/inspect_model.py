import torch

model = torch.jit.load("burgers_unified_model.pt")
print(model.code)
