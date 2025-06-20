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
        self.nu = 0.01 / torch.pi # This nu is only used in the forward pass for loss calculation

    def forward(self, t_collocation: torch.Tensor, x_collocation: torch.Tensor,
                t_initial: torch.Tensor, x_initial: torch.Tensor, u_initial_true: torch.Tensor,
                t_boundary: torch.Tensor, x_boundary: torch.Tensor, u_boundary_true: torch.Tensor) -> torch.Tensor:

        # This forward method is for training and loss calculation
        # For comparison, we will use the .net submodule directly for inference

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

def plot_fortran_trained_model():
    # Define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Trained Model and Perform Python Inference ---
    trained_model_path = "./examples/fortran/burgers/burgers_model_trained.pt"
    
    try:
        # Load the scripted BurgersPINN model (which contains the .net submodule)
        loaded_model = torch.jit.load(trained_model_path, map_location=device)
        loaded_model.eval() # Set to evaluation mode
        print(f"Trained model loaded from {trained_model_path}")

        # Define the spatial and temporal domains (must match Fortran's inference grid setup)
        x_min, x_max = -1.0, 1.0
        t_min, t_max = 0.0, 1.0
        num_points_x = 256
        num_points_t = 100

        x_plot = np.linspace(x_min, x_max, num_points_x)
        t_plot = np.linspace(t_min, t_max, num_points_t)

        # Create meshgrid for (x, t) pairs
        X_grid, T_grid = np.meshgrid(x_plot, t_plot)
        
        t_flat_python = torch.tensor(T_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        x_flat_python = torch.tensor(X_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

        # Perform inference using the .net submodule of the loaded BurgersPINN
        with torch.no_grad():
            u_inference_python_flat = loaded_model.net(torch.cat([t_flat_python, x_flat_python], dim=1)).cpu().numpy()

        U_python = u_inference_python_flat.reshape(num_points_t, num_points_x)
        print(f"Python inference performed. U_python shape: {U_python.shape}")

    except FileNotFoundError:
        print(f"Error: Trained model file not found at {trained_model_path}")
        print("Please ensure burgers03.f90 has been run successfully.")
        return
    except Exception as e:
        print(f"Error during Python model loading or inference: {e}")
        return

    # --- Plotting the Python Inference Result ---
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X_grid, T_grid, U_python, cmap='viridis', shading='auto')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Predicted Solution of 1D Burgers\' Equation (from Fortran Trained Model, Python Inferred)')
    plt.show()

if __name__ == "__main__":
    plot_fortran_trained_model()
