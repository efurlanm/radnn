import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

# Define the BurgersPINN model architecture (must match the one used in generate_burgers_model.py)
class BurgersPINN(torch.nn.Module):
    def __init__(self):
        super(BurgersPINN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 1)
        )
        self.nu = 0.01 / torch.pi

    def forward(self, t_collocation: torch.Tensor, x_collocation: torch.Tensor,
                t_initial: torch.Tensor, x_initial: torch.Tensor, u_initial_true: torch.Tensor,
                t_boundary: torch.Tensor, x_boundary: torch.Tensor, u_boundary_true: torch.Tensor) -> torch.Tensor:

        # PDE loss
        t_collocation_grad = t_collocation.clone().detach().requires_grad_(True)
        x_collocation_grad = x_collocation.clone().detach().requires_grad_(True)
        u_collocation = self.net(torch.cat([t_collocation_grad, x_collocation_grad], dim=1))

        ones_like_u = torch.ones_like(u_collocation)
        grad_outputs_u: List[Optional[torch.Tensor]] = [ones_like_u]

        u_t_opt = torch.autograd.grad([u_collocation], [t_collocation_grad], grad_outputs=grad_outputs_u, create_graph=True)[0]
        u_t = u_t_opt if u_t_opt is not None else torch.zeros_like(u_collocation) # Handle Optional[Tensor]

        u_x_opt = torch.autograd.grad([u_collocation], [x_collocation_grad], grad_outputs=grad_outputs_u, create_graph=True)[0]
        u_x = u_x_opt if u_x_opt is not None else torch.zeros_like(u_collocation) # Handle Optional[Tensor]

        ones_like_ux = torch.ones_like(u_x)
        grad_outputs_ux: List[Optional[torch.Tensor]] = [ones_like_ux]
        u_xx_opt = torch.autograd.grad([u_x], [x_collocation_grad], grad_outputs=grad_outputs_ux, create_graph=True)[0]
        u_xx = u_xx_opt if u_xx_opt is not None else torch.zeros_like(u_x) # Handle Optional[Tensor]

        f = u_t + u_collocation * u_x - self.nu * u_xx
        loss_f = torch.mean(f**2)

        # Initial condition loss
        u_initial_pred = self.net(torch.cat([t_initial, x_initial], dim=1))
        loss_i = torch.mean((u_initial_pred - u_initial_true)**2)

        # Boundary condition loss
        u_boundary_pred = self.net(torch.cat([t_boundary, x_boundary], dim=1))
        loss_b = torch.mean((u_boundary_pred - u_boundary_true)**2)

        return loss_f + loss_i + loss_b

def plot_results():
    # Define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model from the Fortran run
    # The Fortran code saves the entire BurgersPINN model after training
    model_path = "./examples/fortran/burgers/burgers_model_trained.pt"
    
    # Load the scripted model
    loaded_model = torch.jit.load(model_path, map_location=device)
    loaded_model.eval() # Set to evaluation mode

    # Define the spatial and temporal domains
    x_min, x_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0

    # Create a grid for plotting
    num_points_x = 256
    num_points_t = 100
    x_plot = np.linspace(x_min, x_max, num_points_x)
    t_plot = np.linspace(t_min, t_max, num_points_t)

    # Create meshgrid for (x, t) pairs
    X, T = np.meshgrid(x_plot, t_plot)
    
    # Flatten and convert to tensors for model input
    # The inference model (loaded_model.net) expects t and x as separate inputs
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

    # Perform inference using the .net submodule of the loaded BurgersPINN
    with torch.no_grad(): # No need to compute gradients for inference
        u_pred_flat = loaded_model.net(torch.cat([t_flat, x_flat], dim=1)).cpu().numpy()

    # Reshape the predictions back to the grid shape
    U_pred = u_pred_flat.reshape(num_points_t, num_points_x)

    # Plotting (similar to burgers1d-pytorch.ipynb)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, T, U_pred, cmap='viridis', shading='auto')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Predicted Solution of 1D Burgers\' Equation (from Fortran Trained Model)')
    plt.show()

if __name__ == "__main__":
    plot_results()
