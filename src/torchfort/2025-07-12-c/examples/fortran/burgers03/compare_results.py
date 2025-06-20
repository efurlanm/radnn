import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the original Python inference results
with open("burgers1d_python_original_results.bin", 'rb') as f:
    N_x_orig, N_t_orig = np.fromfile(f, dtype=np.int32, count=2)
    x_orig = np.fromfile(f, dtype=np.float32, count=N_x_orig)
    t_orig = np.fromfile(f, dtype=np.float32, count=N_t_orig)
    u_pred_orig = np.fromfile(f, dtype=np.float32).reshape(N_t_orig, N_x_orig)

# Load the Fortran-trained model
model = torch.jit.load("burgers_inference_trained.pt")
model.eval()

# Generate a grid for visualization
X, T = np.meshgrid(x_orig, t_orig)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
XT_tensor = torch.tensor(XT, dtype=torch.float32)

# Perform inference with the Fortran-trained model
with torch.no_grad():
    u_pred_fortran = model(XT_tensor).numpy().reshape(N_t_orig, N_x_orig)

# Compare the results
abs_diff = np.abs(u_pred_orig - u_pred_fortran)
max_abs_diff = np.max(abs_diff)
mean_abs_diff = np.mean(abs_diff)

print(f"Maximum absolute difference: {max_abs_diff:.2e}")
print(f"Mean absolute difference: {mean_abs_diff:.2e}")

# Plot the results for comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original Python results
cf1 = axes[0].contourf(X, T, u_pred_orig, levels=100, cmap='viridis')
fig.colorbar(cf1, ax=axes[0], label='u(x,t)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].set_title('Original Python PINN')

# Fortran-trained model results
cf2 = axes[1].contourf(X, T, u_pred_fortran, levels=100, cmap='viridis')
fig.colorbar(cf2, ax=axes[1], label='u(x,t)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
axes[1].set_title('Fortran-Trained PINN')

# Absolute difference
cf3 = axes[2].contourf(X, T, abs_diff, levels=100, cmap='Reds')
fig.colorbar(cf3, ax=axes[2], label='Absolute Difference')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
axes[2].set_title('Absolute Difference')

plt.tight_layout()
plt.show()
