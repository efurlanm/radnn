import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Fixed random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set the default tensor type to float32 for compatibility with numerical operations
torch.set_default_dtype(torch.float32)

# Define physical parameters and boundaries
x_min, x_max = -1.0, 1.0  # Spatial range
t_min, t_max = 0.0, 1.0   # Temporal range
nu = 0.01 / np.pi         # Diffusion coefficient

# Define the number of sampled points
N_f = 10000  # Collocation points for the partial differential equation (PDE)
N_0 = 400    # Initial condition points
N_b = 200    # Boundary condition points

# Generate collocation points for training the PINN
X_f = np.random.rand(N_f, 2)
X_f[:, 0] = X_f[:, 0] * (x_max - x_min) + x_min
X_f[:, 1] = X_f[:, 1] * (t_max - t_min) + t_min

# Initial condition u(x, 0) = -sin(pi * x)
x0 = np.linspace(x_min, x_max, N_0)[:, None]
t0 = np.zeros_like(x0)
u0 = -np.sin(np.pi * x0)

# Boundary conditions
tb = np.linspace(t_min, t_max, N_b)[:, None]
xb_left = np.ones_like(tb) * x_min
xb_right = np.ones_like(tb) * x_max

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to PyTorch tensors
X_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(device)
x0_t0 = torch.tensor(np.hstack((x0, t0)), dtype=torch.float32).to(device)
u0 = torch.tensor(u0, dtype=torch.float32).to(device)
xb_left_tb = torch.tensor(np.hstack((xb_left, tb)), dtype=torch.float32).to(device)
xb_right_tb = torch.tensor(np.hstack((xb_right, tb)), dtype=torch.float32).to(device)

# Define the PINN neural network
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# Function to compute the PDE residual
def pde_residual(model, X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    u = model(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = u_t + u * u_x - nu * u_xx
    return f

# Loss function
def loss_func(model):
    loss_f = torch.mean(pde_residual(model, X_f)**2)
    loss_0 = torch.mean((model(x0_t0) - u0)**2)
    loss_b = torch.mean(model(xb_left_tb)**2) + torch.mean(model(xb_right_tb)**2)
    return loss_f + loss_0 + loss_b

# Initialize the model and optimizer
layers = [2, 10, 10, 10, 10, 10, 10, 10, 10, 1]
model = PINN(layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_func(model)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 500 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.5e}')

print("Training complete!")
model.eval()

# Generate a grid for visualization and save results
N_x, N_t = 256, 100
x = np.linspace(x_min, x_max, N_x)
t = np.linspace(t_min, t_max, N_t)
X, T = np.meshgrid(x, t)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
XT_tensor = torch.tensor(XT, dtype=torch.float32).to(device)

with torch.no_grad():
    u_pred = model(XT_tensor).cpu().numpy().reshape(N_t, N_x)

output_filename = "burgers1d_python_original_results.bin"
with open(output_filename, 'wb') as f:
    f.write(np.array([N_x, N_t], dtype=np.int32).tobytes())
    f.write(x.astype(np.float32).tobytes())
    f.write(t.astype(np.float32).tobytes())
    f.write(u_pred.astype(np.float32).tobytes())
print(f"Original Python inference results saved to {output_filename}")