import torch
import torch.nn as nn
import numpy as np
from typing import List

# This is the simple network that predicts u(x, t)
# It will be loaded as the main model in Fortran.
class InferenceNet(nn.Module):
    def __init__(self):
        super(InferenceNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

@torch.jit.script
def compute_gradients(u, x) -> torch.Tensor:
    grad = torch.autograd.grad([u], [x], create_graph=True, allow_unused=True)[0]
    if grad is None:
        raise RuntimeError("Gradient computation failed")
    return grad

# This module computes the total PINN loss.
# It will be loaded as the loss function in Fortran.
class LossPINN(nn.Module):
    def __init__(self, nu: float, inference_net: InferenceNet):
        super(LossPINN, self).__init__()
        self.nu = nu
        self.inference_net = inference_net

    def forward(self, X_f, x0_t0, u0, xb_left_tb, xb_right_tb):
        # This forward method accepts the 5 tensors from Fortran and computes the total loss.
        
        # 1. PDE Residual Loss
        X_f.requires_grad_(True)
        u_f = self.inference_net(X_f)
        
        grad_u = compute_gradients(u_f, X_f)
        u_t = grad_u[:, 1:2]
        u_x = grad_u[:, 0:1]
        
        grad_u_x = compute_gradients(u_x, X_f)
        u_xx = grad_u_x[:, 0:1]
        
        f = u_t + u_f * u_x - self.nu * u_xx
        loss_f = torch.mean(f**2)

        # 2. Initial Condition Loss
        u_pred_0 = self.inference_net(x0_t0)
        loss_0 = torch.mean((u_pred_0 - u0)**2)

        # 3. Boundary Condition Loss
        u_pred_b_left = self.inference_net(xb_left_tb)
        u_pred_b_right = self.inference_net(xb_right_tb)
        loss_b = torch.mean(u_pred_b_left**2) + torch.mean(u_pred_b_right**2)

        return loss_f + loss_0 + loss_b

# --- Main script ---
nu = 0.01 / np.pi
inference_model = InferenceNet()
loss_model = LossPINN(nu=nu, inference_net=inference_model)

# Script and save both models
scripted_inference_model = torch.jit.script(inference_model)
scripted_loss_model = torch.jit.script(loss_model)

scripted_inference_model.save("inference_net.pt")
scripted_loss_model.save("loss_model.pt")

print("Successfully generated inference_net.pt and loss_model.pt")