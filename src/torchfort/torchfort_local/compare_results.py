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

    # --- Load Original Python Inference Results ---
    original_python_results_path = "burgers1d_python_original_results.bin"
    
    try:
        with open(original_python_results_path, 'rb') as f:
            num_points_x_orig = np.fromfile(f, dtype=np.int32, count=1)[0]
            num_points_t_orig = np.fromfile(f, dtype=np.int32, count=1)[0]
            x_plot_orig = np.fromfile(f, dtype=np.float32, count=num_points_x_orig)
            t_plot_orig = np.fromfile(f, dtype=np.float32, count=num_points_t_orig)
            u_inference_orig_flat = np.fromfile(f, dtype=np.float32, count=num_points_x_orig * num_points_t_orig)
        
        U_original_python = u_inference_orig_flat.reshape(num_points_t_orig, num_points_x_orig)
        print(f"Original Python results loaded from {original_python_results_path}")
        print(f"Original Python U_pred shape: {U_original_python.shape}")

    except FileNotFoundError:
        print(f"Error: Original Python results file not found at {original_python_results_path}")
        print("Please run burgers1d.py first to generate this file.")
        return

    # --- Load Fortran-Trained Model and Perform Python Inference ---
    fortran_trained_model_path = "./examples/fortran/burgers/burgers_model_trained.pt"
    
    try:
        # Load the scripted BurgersPINN model (which contains the .net submodule)
        loaded_model = torch.jit.load(fortran_trained_model_path, map_location=device)
        loaded_model.eval() # Set to evaluation mode
        print(f"Fortran-trained model loaded from {fortran_trained_model_path}")

        # Use the same grid points as original Python for inference
        X_grid_fortran_trained, T_grid_fortran_trained = np.meshgrid(x_plot_orig, t_plot_orig)
        
        t_flat_fortran_trained = torch.tensor(T_grid_fortran_trained.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        x_flat_fortran_trained = torch.tensor(X_grid_fortran_trained.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

        # Perform inference using the .net submodule of the loaded BurgersPINN
        with torch.no_grad():
            u_inference_fortran_trained_flat = loaded_model.net(torch.cat([t_flat_fortran_trained, x_flat_fortran_trained], dim=1)).cpu().numpy()

        U_fortran_trained = u_inference_fortran_trained_flat.reshape(num_points_t_orig, num_points_x_orig)
        print(f"Inference with Fortran-trained model performed. U_fortran_trained shape: {U_fortran_trained.shape}")

    except FileNotFoundError:
        print(f"Error: Fortran-trained model file not found at {fortran_trained_model_path}")
        print("Please ensure burgers03.f90 has been run successfully.")
        return
    except Exception as e:
        print(f"Error during Fortran-trained model loading or inference: {e}")
        return

    # --- Compare Results ---
    if U_original_python.shape != U_fortran_trained.shape:
        print("Error: Shapes of Original Python and Fortran-trained Python inference results do not match!")
        print(f"Original Python shape: {U_original_python.shape}, Fortran-trained shape: {U_fortran_trained.shape}")
        return

    abs_diff = np.abs(U_original_python - U_fortran_trained)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    print(f"\nComparison Results (Original Python vs. Fortran-trained Python):")
    print(f"  Maximum Absolute Difference: {max_abs_diff:.5e}")
    print(f"  Mean Absolute Difference: {mean_abs_diff:.5e}")

    # --- Plotting the Difference ---
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X_grid_fortran_trained, T_grid_fortran_trained, abs_diff, cmap='hot', shading='auto')
    plt.colorbar(label='Absolute Difference |u_original - u_fortran_trained|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Absolute Difference: Original Python vs. Fortran-trained Model')
    plt.show()

    # --- Optional: Plot side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Original Python Plot
    im1 = axes[0].pcolormesh(x_plot_orig, t_plot_orig, U_original_python, cmap='viridis', shading='auto')
    fig.colorbar(im1, ax=axes[0], label='u(x,t)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('Original Python Inference Result')

    # Fortran-trained Python Plot
    im2 = axes[1].pcolormesh(x_plot_orig, t_plot_orig, U_fortran_trained, cmap='viridis', shading='auto')
    fig.colorbar(im2, ax=axes[1], label='u(x,t)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('Fortran-trained Model Inference Result')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_results()
