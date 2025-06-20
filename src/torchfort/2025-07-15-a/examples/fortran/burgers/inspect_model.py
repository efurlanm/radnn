import torch

model = torch.jit.load("burgers_model_trained.pt")
print(model.code)
