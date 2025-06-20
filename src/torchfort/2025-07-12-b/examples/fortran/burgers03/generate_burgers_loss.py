import torch
import torch.nn as nn
from typing import Tuple

class BurgersLossModule(nn.Module):
    def __init__(self):
        super().__init__()

    # The model returns 7 tensors. The labels list contains 1 tensor.
    # TorchFort unpacks these into individual arguments.
    def forward(self, u_pred_f: torch.Tensor, u_pred_0: torch.Tensor, 
                u_pred_b_left: torch.Tensor, u_pred_b_right: torch.Tensor, 
                f: torch.Tensor, u_x: torch.Tensor, u_xx: torch.Tensor,
                u0: torch.Tensor) -> torch.Tensor:

        loss_f = torch.mean(f**2)
        loss_0 = torch.mean((u_pred_0 - u0)**2)
        loss_b = torch.mean(u_pred_b_left**2) + torch.mean(u_pred_b_right**2)

        return loss_f + loss_0 + loss_b

loss_module = BurgersLossModule()
scripted_loss_module = torch.jit.script(loss_module)
scripted_loss_module.save("burgers_loss.pt")
print("Scripted Burgers Loss Module saved to burgers_loss.pt")