import torch
import torch.nn as nn

class EmptyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0) # Return a dummy loss

scripted_loss = torch.jit.script(EmptyLoss())
scripted_loss.save("burgers_loss.pt")