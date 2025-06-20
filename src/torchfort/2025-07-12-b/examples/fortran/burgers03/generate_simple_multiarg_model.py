import torch
import torch.nn as nn
from typing import Tuple

class SimpleMultiArgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # This model takes two inputs and returns two outputs
        out1 = self.linear1(x1)
        out2 = self.linear2(x2)
        return out1, out2

model = SimpleMultiArgModel()
scripted_model = torch.jit.script(model)
scripted_model.save("simple_multiarg_model.pt")
print("Simple multi-argument model saved to simple_multiarg_model.pt")
