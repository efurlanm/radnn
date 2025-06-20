import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net_layers = []
        for i in range(len(layers) - 2):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            net_layers.append(nn.Tanh())
        net_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.net(x)

# Define physical parameters and boundaries (must match training script)
x_min, x_max = -1.0, 1.0  # Spatial range
t_min, t_max = 0.0, 1.0   # Temporal range

# Define the number of sampled points for inference grid
N_x, N_t = 256, 100

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move to GPU if available
layers = [2, 10, 10, 10, 10, 10, 10, 10, 10, 1]  # Must match training architecture
model = PINN(layers).to(device)
model.load_state_dict(torch.load("pinn_model_state.pt"))
model.eval()

# Generate a grid for visualization (must match original Python script)
x = np.linspace(x_min, x_max, N_x)
t = np.linspace(t_min, t_max, N_t)
X, T = np.meshgrid(x, t)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
XT_tensor = torch.tensor(XT, dtype=torch.float32).to(device)

# Perform inference
with torch.no_grad():
    u_pred = model(XT_tensor).cpu().numpy().reshape(N_t, N_x)

# Save the predicted solution to a text file
output_filename = "python_inference_u_pred.txt"
np.savetxt(output_filename, u_pred)
print(f"Python inference results saved to {output_filename}")

# Plot the predicted solution
plt.figure(figsize=(8, 5))
plt.contourf(X, T, u_pred, levels=100, cmap='viridis')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title("Predicted solution u(x,t) via PINN (Python Inference)")
plt.show()