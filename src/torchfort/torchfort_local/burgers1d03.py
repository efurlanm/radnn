import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set the default tensor type to float32 for compatibility with numerical operations
torch.set_default_dtype(torch.float32)

# Define physical parameters and boundaries
x_min, x_max = -1.0, 1.0  # Spatial range
t_min, t_max = 0.0, 1.0   # Temporal range
nu = 0.01 / np.pi         # Diffusion coefficient

# Define the number of sampled points
N_f = 10000  # Collocation points for the partial differential equation (PDE)
N_0 = 200    # Initial condition points
N_b = 200    # Boundary condition points

# Generate collocation points for training the PINN
X_f = np.random.rand(N_f, 2)  # Generate random points in a 2D space
X_f[:, 0] = X_f[:, 0] * (x_max - x_min) + x_min  # Normalize x to [-1, 1]
X_f[:, 1] = X_f[:, 1] * (t_max - t_min) + t_min  # Normalize t to [0, 1]

# Initial condition u(x, 0) = -sin(pi * x)
x0 = np.linspace(x_min, x_max, N_0)[:, None]  # Discretized spatial domain
t0 = np.zeros_like(x0)  # Fixed initial time (t = 0)
u0 = -np.sin(np.pi * x0)  # Sine function applied to initial condition

# Boundary conditions u(-1,t) = 0 and u(1,t) = 0
tb = np.linspace(t_min, t_max, N_b)[:, None]  # Time points for boundary
xb_left = np.ones_like(tb) * x_min  # Left boundary position (x = -1)
xb_right = np.ones_like(tb) * x_max # Right boundary position (x = 1)
ub_left = np.zeros_like(tb)  # Function value at the left boundary
ub_right = np.zeros_like(tb) # Function value at the right boundary

# Check if a GPU is available to accelerate computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to PyTorch tensors and move them to GPU if available
X_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(device)
x0 = torch.tensor(x0, dtype=torch.float32).to(device)
t0 = torch.tensor(t0, dtype=torch.float32).to(device)
u0 = torch.tensor(u0, dtype=torch.float32).to(device)
tb = torch.tensor(tb, dtype=torch.float32).to(device)
xb_left = torch.tensor(xb_left, dtype=torch.float32).to(device)
xb_right = torch.tensor(xb_right, dtype=torch.float32).to(device)
ub_left = torch.tensor(ub_left, dtype=torch.float32).to(device)
ub_right = torch.tensor(ub_right, dtype=torch.float32).to(device)
# Define the PINN neural network using PyTorch
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()  # Non-linear activation function
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        """Defines the flow of data through the neural network."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))  # Apply activation function in hidden layers
        return self.layers[-1](x)  # Final layer without activation (output u(x,t))

# Initialize the model and move to GPU if available
layers = [2, 50, 50, 50, 50, 1]  # Network structure: input (x, t), 4 hidden layers, output u(x,t)
model = PINN(layers).to(device)
# Function to compute the residual of the partial differential equation using autograd
def pde_residual(model, X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    u = model(torch.cat([x, t], dim=1))

    # Compute derivatives using PyTorch's autograd mechanism
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # PDE residual: u_t + u * u_x - nu * u_xx = 0
    f = u_t + u * u_x - nu * u_xx
    return f
# Loss function combining multiple constraints
def loss_func(model):
    loss_f = torch.mean(pde_residual(model, X_f)**2)  # PDE loss
    loss_0 = torch.mean((model(torch.cat([x0, t0], dim=1)) - u0)**2)  # Initial condition loss
    loss_b = torch.mean(model(torch.cat([xb_left, tb], dim=1))**2) + torch.mean(model(torch.cat([xb_right, tb], dim=1))**2)  # Boundary condition loss
    return loss_f + loss_0 + loss_b
# Configure the Adam optimizer for adjusting the network's weights
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop for the PINN
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset accumulated gradients
    loss = loss_func(model)  # Compute the total loss
    loss.backward()  # Compute gradients via backpropagation
    optimizer.step()  # Update parameters based on computed gradients

    # Print loss every 500 epochs to track progress
    if (epoch+1) % 500 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.5e}')

print("Training complete!")
# Evaluate the model
model.eval()

# Generate a grid for visualization
N_x, N_t = 256, 100
x = np.linspace(x_min, x_max, N_x)
t = np.linspace(t_min, t_max, N_t)
X, T = np.meshgrid(x, t)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
XT_tensor = torch.tensor(XT, dtype=torch.float32).to(device)

# Compute the predicted solution using the trained model
with torch.no_grad():
    u_pred = model(XT_tensor).cpu().numpy().reshape(N_t, N_x)

# Save results to a binary file
output_filename = "burgers1d_python_original_results.bin"
with open(output_filename, 'wb') as f:
    f.write(np.array([N_x, N_t], dtype=np.int32).tobytes())
    f.write(x.astype(np.float32).tobytes())
    f.write(t.astype(np.float32).tobytes())
    f.write(u_pred.astype(np.float32).tobytes())
print(f"Original Python inference results saved to {output_filename}")

# Plot the predicted solution
plt.figure(figsize=(8, 5))
plt.contourf(X, T, u_pred, levels=100, cmap='viridis')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title("Predicted solution u(x,t) via PINN")
plt.show()