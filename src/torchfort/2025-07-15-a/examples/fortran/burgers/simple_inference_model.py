import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1) # Input (x,t), Output (u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

scripted_model = torch.jit.script(SimpleNet())
scripted_model.save("simple_inference_model.pt")
print("simple_inference_model.pt created.")