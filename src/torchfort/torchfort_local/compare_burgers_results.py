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

def compare_results():
    # Define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Fortran Inference Results ---
    fortran_results_path = "./examples/fortran/burgers/burgers_inference_results.bin"
    
    try:
        with open(fortran_results_path, 'rb') as f:
            num_points_x_fortran = np.fromfile(f, dtype=np.int32, count=1)[0]
            num_points_t_fortran = np.fromfile(f, dtype=np.int32, count=1)[0]
            x_plot_fortran = np.fromfile(f, dtype=np.float32, count=num_points_x_fortran)
            t_plot_fortran = np.fromfile(f, dtype=np.float32, count=num_points_t_fortran)
            u_inference_fortran_flat = np.fromfile(f, dtype=np.float32, count=num_points_x_fortran * num_points_t_fortran)
        
        # Reshape Fortran results to (num_t, num_x) for plotting
        # Fortran saved u_inference as (1, num_points_x * num_points_t) in column-major order.
        # When read into numpy, it's a 1D array. Reshape to (num_x, num_t) then transpose.
        U_fortran = u_inference_fortran_flat.reshape(num_points_x_fortran, num_points_t_fortran).T
        print(f"Fortran results loaded from {fortran_results_path}")
        print(f"Fortran U_pred shape: {U_fortran.shape}")

    except FileNotFoundError:
        print(f"Error: Fortran results file not found at {fortran_results_path}")
        print("Please ensure burgers03.f90 has been run successfully.")
        return

    # --- Load Trained Model and Perform Python Inference ---
    trained_model_path = "./examples/fortran/burgers/burgers_model_trained.pt"
    
    try:
        # Load the scripted BurgersPINN model (which contains the .net submodule)
        loaded_model = torch.jit.load(trained_model_path, map_location=device)
        loaded_model.eval() # Set to evaluation mode
        print(f"Trained model loaded from {trained_model_path}")

        # Use the same grid points as Fortran for Python inference
        X_python, T_python = np.meshgrid(x_plot_fortran, t_plot_fortran)
        
        t_flat_python = torch.tensor(T_python.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        x_flat_python = torch.tensor(X_python.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

        # Perform inference using the .net submodule of the loaded BurgersPINN
        with torch.no_grad():
            u_inference_python_flat = loaded_model.net(torch.cat([t_flat_python, x_flat_python], dim=1)).cpu().numpy()

        U_python = u_inference_python_flat.reshape(num_points_t_fortran, num_points_x_fortran)
        print(f"Python inference performed. U_python shape: {U_python.shape}")

    except FileNotFoundError:
        print(f"Error: Trained model file not found at {trained_model_path}")
        print("Please ensure burgers03.f90 has been run successfully.")
        return
    except Exception as e:
        print(f"Error during Python model loading or inference: {e}")
        return

    # --- Compare Results ---
    if U_fortran.shape != U_python.shape:
        print("Error: Shapes of Fortran and Python inference results do not match!")
        print(f"Fortran shape: {U_fortran.shape}, Python shape: {U_python.shape}")
        return

    abs_diff = np.abs(U_fortran - U_python)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    print(f"\nComparison Results:")
    print(f"  Maximum Absolute Difference: {max_abs_diff:.5e}")
    print(f"  Mean Absolute Difference: {mean_abs_diff:.5e}")

    # --- Plotting the Difference ---
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X_python, T_python, abs_diff, cmap='hot', shading='auto')
    plt.colorbar(label='Absolute Difference |u_fortran - u_python|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Absolute Difference between Fortran and Python Inference Results')
    plt.show()

    # --- Optional: Plot Fortran and Python results side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Fortran Plot
    im1 = axes[0].pcolormesh(x_plot_fortran, t_plot_fortran, U_fortran, cmap='viridis', shading='auto')
    fig.colorbar(im1, ax=axes[0], label='u(x,t)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('Fortran Inference Result')

    # Python Plot
    im2 = axes[1].pcolormesh(x_plot_fortran, t_plot_fortran, U_python, cmap='viridis', shading='auto')
    fig.colorbar(im2, ax=axes[1], label='u(x,t)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('Python Inference Result')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_results()