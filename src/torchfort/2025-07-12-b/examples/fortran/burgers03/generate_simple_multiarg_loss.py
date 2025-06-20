import torch
import torch.nn as nn
from typing import Tuple

class SimpleMultiArgLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_outputs: Tuple[torch.Tensor, torch.Tensor], labels: Tuple[torch.Tensor]) -> torch.Tensor:
        # model_outputs will be (out1, out2) from SimpleMultiArgModel
        # labels will be (target1,)
        out1, out2 = model_outputs
        target1 = labels[0]

        loss = torch.mean((out1 - target1)**2) + torch.mean(out2**2)
        return loss

loss_module = SimpleMultiArgLoss()
scripted_loss_module = torch.jit.script(loss_module)
scripted_loss_module.save("simple_multiarg_loss.pt")
print("Simple multi-argument loss saved to simple_multiarg_loss.pt")
