import torch
import torch.nn as nn
import numpy as np
from typing import List

# Set the default tensor type to float32
torch.set_default_dtype(torch.float32)

# Define physical parameters
nu = 0.01 / np.pi

# This is the unified module we will script and use in Fortran.
# It contains the PINN model and the loss calculation logic.
class BurgersPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.nu = nu
        
        # Define the neural network layers
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

    def pinn_forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is the core neural network forward pass
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

    def forward(self, inputs: List[torch.Tensor], labels: List[torch.Tensor]) -> torch.Tensor:
        # This forward method calculates the loss for training.
        # It takes the tensors as lists of inputs and labels.

        X_f = inputs[0]
        x0_t0 = inputs[1]
        xb_left_tb = inputs[2]
        xb_right_tb = inputs[3]
        u0 = labels[0]

        # 1. PDE Residual Loss
        X_f.requires_grad_(True)
        u_pred_f = self.pinn_forward(X_f)
        
        # Compute derivatives
        grad_outputs = torch.ones_like(u_pred_f)
        grad_tuple = torch.autograd.grad(
            (u_pred_f,), (X_f,), grad_outputs=(grad_outputs,), create_graph=True
        )
        grads = grad_tuple[0]
        assert grads is not None, "First derivative calculation failed"
        u_t = grads[:, 1:2]
        u_x = grads[:, 0:1]
        
        # Compute second derivative
        grad_outputs_x = torch.ones_like(u_x)
        grad_tuple_2 = torch.autograd.grad(
            (u_x,), (X_f,), grad_outputs=(grad_outputs_x,), create_graph=True
        )
        u_xx_grads = grad_tuple_2[0]
        assert u_xx_grads is not None, "Second derivative calculation failed"
        u_xx = u_xx_grads[:, 0:1]

        f = u_t + u_pred_f * u_x - self.nu * u_xx
        loss_f = torch.mean(f**2)

        # 2. Initial Condition Loss
        u_pred_0 = self.pinn_forward(x0_t0)
        loss_0 = torch.mean((u_pred_0 - u0)**2)

        # 3. Boundary Condition Loss
        u_pred_b_left = self.pinn_forward(xb_left_tb)
        u_pred_b_right = self.pinn_forward(xb_right_tb)
        loss_b = torch.mean(u_pred_b_left**2) + torch.mean(u_pred_b_right**2)

        # The 'forward' function for a training model in TorchFort must return the loss
        return loss_f + loss_0 + loss_b

# Instantiate the unified model
burgers_model = BurgersPINN()

# Script the unified model and save it. This single file will be used for training.
# Define dummy inputs for scripting the unified model
N_f = 10000
N_0 = 400
N_b = 200

dummy_X_f = torch.randn(N_f, 2, dtype=torch.float32)
dummy_x0_t0 = torch.randn(N_0, 2, dtype=torch.float32)
dummy_xb_left_tb = torch.randn(N_b, 2, dtype=torch.float32)
dummy_xb_right_tb = torch.randn(N_b, 2, dtype=torch.float32)
dummy_u0 = torch.randn(N_0, 1, dtype=torch.float32)

dummy_inputs = [dummy_X_f, dummy_x0_t0, dummy_xb_left_tb, dummy_xb_right_tb]
dummy_labels = [dummy_u0]

scripted_model = torch.jit.script(burgers_model, (dummy_inputs, dummy_labels))
scripted_model.save("burgers_model.pt")
print("Scripted unified training model saved to burgers_model.pt")

# For inference, we create a separate class that only has the network pass.
# This makes it cleaner to use after training.
class BurgersInferenceNet(nn.Module):
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

# Instantiate the inference network
inference_net = BurgersInferenceNet()
# We can load the initial (untrained) weights from our unified model
inference_net.load_state_dict(burgers_model.state_dict())
# Script and save the inference-only model.
# The Fortran code will save the trained weights to a new file, which can then be loaded into this network.
scripted_inference_net = torch.jit.script(inference_net)
scripted_inference_net.save("burgers_inference_net.pt")
print("Scripted inference-only network saved to burgers_inference_net.pt")
