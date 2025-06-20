import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

torch.set_default_dtype(torch.float32)

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net_layers = []
        for i in range(len(layers) - 2):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            net_layers.append(nn.Tanh())
        net_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net_layers)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)

class BurgersPINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.inference_net = PINN(layers)
        self.register_buffer('nu', torch.tensor(0.01 / np.pi))

    def forward(self, x_f: torch.Tensor, x0_cat: torch.Tensor, 
                xb_left_cat: torch.Tensor, xb_right_cat: torch.Tensor) -> torch.Tensor:
        """
        This forward method returns predictions and residuals, but does not take u_0.
        It expects concatenated input tensors for initial and boundary conditions.
        """
        # Predictions for all point types
        u_0_pred = self.inference_net(x0_cat)
        u_b_left_pred = self.inference_net(xb_left_cat)
        u_b_right_pred = self.inference_net(xb_right_cat)

        # PDE residual calculation requires gradients
        x_f_clone = x_f.clone()
        x_f_clone.requires_grad_(True)
        u_f_pred = self.inference_net(x_f_clone) # Pass the concatenated (x,t) tensor directly

        u_grads_tuple = torch.autograd.grad(outputs=(u_f_pred,), inputs=(x_f_clone,), grad_outputs=(torch.ones_like(u_f_pred),), create_graph=True, retain_graph=True)
        u_grads = u_grads_tuple[0]
        if u_grads is None:
            raise RuntimeError("Failed to compute gradients for u_t and u_x")
        u_t = u_grads[:, 1:2]
        u_x = u_grads[:, 0:1]

        u_xx_tuple = torch.autograd.grad(outputs=(u_x,), inputs=(x_f_clone,), grad_outputs=(torch.ones_like(u_x),), create_graph=True)
        u_xx_grad = u_xx_tuple[0]
        if u_xx_grad is None:
            raise RuntimeError("Failed to compute gradients for u_xx")
        u_xx = u_xx_grad[:, 0:1]

        pde_residual = u_t + u_f_pred * u_x - self.nu * u_xx

        return torch.cat((u_0_pred, u_b_left_pred, u_b_right_pred, pde_residual), dim=0)

class BurgersLoss(nn.Module):
    def __init__(self, N_0, N_b, N_f):
        super().__init__()
        self.N_0 = N_0
        self.N_b = N_b
        self.N_f = N_f

    def forward(self, predictions: torch.Tensor, u_0_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the total loss from the model's predictions and the target data.
        """
        u_0_pred = predictions[0:self.N_0]
        u_b_left_pred = predictions[self.N_0:self.N_0+self.N_b]
        u_b_right_pred = predictions[self.N_0+self.N_b:self.N_0+2*self.N_b]
        pde_residual = predictions[self.N_0+2*self.N_b:]
        
        loss_0 = torch.mean((u_0_pred - u_0_target)**2)
        loss_b = torch.mean(u_b_left_pred**2) + torch.mean(u_b_right_pred**2)
        loss_f = torch.mean(pde_residual**2)
        return loss_0 + loss_b + loss_f

def main():
    layers = [2, 10, 10, 10, 10, 10, 10, 10, 10, 1] # Using layers from burgers1d.py
    N_f = 10000
    N_0 = 400 # Using N_0 from burgers1d.py
    N_b = 200 # Using N_b from burgers1d.py
    burgers_model = BurgersPINN(layers)
    loss_fn = BurgersLoss(N_0, N_b, N_f)

    scripted_burgers_model = torch.jit.script(burgers_model)
    scripted_loss_fn = torch.jit.script(loss_fn)

    model_filename = "burgers_model.pt"
    loss_filename = "burgers_loss.pt"
    scripted_burgers_model.save(model_filename)
    scripted_loss_fn.save(loss_filename)

    print(f"Model saved to {model_filename}")
    print(f"Loss function saved to {loss_filename}")

if __name__ == '__main__':
    main()