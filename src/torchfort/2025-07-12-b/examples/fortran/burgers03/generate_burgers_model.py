import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Define physical parameters
nu = 0.01 / np.pi

# Define the core PINN neural network
class BurgersPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc_out = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose is handled in Fortran now
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        x = self.activation(self.fc7(x))
        x = self.activation(self.fc8(x))
        x = self.fc_out(x)
        return x

# Define a wrapper module that computes predictions and derivatives
class BurgersModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.pinn = BurgersPINN()
        self.nu = nu

    def forward(self, X_f: torch.Tensor, x0_t0: torch.Tensor, xb_left_tb: torch.Tensor, xb_right_tb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_f.requires_grad_(True)
        u_pred_f = self.pinn(X_f)
        
        grad_outputs = torch.ones_like(u_pred_f)
        grad_tuple = torch.autograd.grad((u_pred_f,), (X_f,), grad_outputs=(grad_outputs,), create_graph=True)
        grads = grad_tuple[0]
        assert grads is not None
        u_t = grads[:, 1:2]
        u_x = grads[:, 0:1]
        
        grad_outputs_x = torch.ones_like(u_x)
        grad_tuple_2 = torch.autograd.grad((u_x,), (X_f,), grad_outputs=(grad_outputs_x,), create_graph=True)
        u_xx_grads = grad_tuple_2[0]
        assert u_xx_grads is not None
        u_xx = u_xx_grads[:, 0:1]

        f = u_t + u_pred_f * u_x - self.nu * u_xx
        
        u_pred_0 = self.pinn(x0_t0)
        u_pred_b_left = self.pinn(xb_left_tb)
        u_pred_b_right = self.pinn(xb_right_tb)

        return u_pred_f, u_pred_0, u_pred_b_left, u_pred_b_right, f, u_x, u_xx

model_wrapper = BurgersModelWrapper()
scripted_model_wrapper = torch.jit.script(model_wrapper)
scripted_model_wrapper.save("burgers_model.pt")
print("Scripted Burgers Model Wrapper saved to burgers_model.pt")

inference_module = BurgersPINN()
inference_module.load_state_dict(model_wrapper.pinn.state_dict())
scripted_inference_module = torch.jit.script(inference_module)
scripted_inference_module.save("burgers_inference_net.pt")
print("Scripted inference network saved to burgers_inference_net.pt")