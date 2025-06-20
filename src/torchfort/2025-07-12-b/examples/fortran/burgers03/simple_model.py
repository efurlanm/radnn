import torch
import torch.nn as nn
import torch.nn as nn
from typing import List

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, e: torch.Tensor, f: torch.Tensor):
        print("Python received 6 arguments.")
        print(f"  Arg 0 (a): shape={a.shape}")
        print(f"  Arg 1 (b): shape={b.shape}")
        print(f"  Arg 2 (c): shape={c.shape}")
        print(f"  Arg 3 (d): shape={d.shape}")
        print(f"  Arg 4 (e): shape={e.shape}")
        print(f"  Arg 5 (f): shape={f.shape}")
        
        # Return a dummy loss
        return torch.tensor(0.0, requires_grad=True)

model = SimpleModel()
scripted_model = torch.jit.script(model)
scripted_model.save("simple_model.pt")
print("Saved simple_model.pt")
