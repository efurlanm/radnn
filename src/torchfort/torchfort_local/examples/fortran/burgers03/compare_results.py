import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the Fortran-trained model
fortran_model = torch.jit.load("burgers_model_trained.pt")
fortran_model.eval()

# Load the original Python results
with open("burgers1d_python_original_results.bin", 'rb') as f:
    N_x, N_t = np.fromfile(f, dtype=np.int32, count=2)
    x = np.fromfile(f, dtype=np.float32, count=N_x)
    t = np.fromfile(f, dtype=np.float32, count=N_t)
    u_pred_python = np.fromfile(f, dtype=np.float32).reshape(N_t, N_x)

# Perform inference with the Fortran-trained model
X, T = np.meshgrid(x, t)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
XT_tensor = torch.tensor(XT, dtype=torch.float32)
with torch.no_grad():
    u_pred_fortran = fortran_model(XT_tensor).numpy().reshape(N_t, N_x)

# Compare the results
abs_diff = np.abs(u_pred_python - u_pred_fortran)
print(f"Maximum absolute difference: {np.max(abs_diff):.2e}")
print(f"Mean absolute difference:    {np.mean(abs_diff):.2e}")

# Plot the results for visual comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Python result
cf1 = axes[0].contourf(X, T, u_pred_python, levels=100, cmap='viridis')
fig.colorbar(cf1, ax=axes[0])
axes[0].set_title('Original Python PINN')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')

# Fortran result
cf2 = axes[1].contourf(X, T, u_pred_fortran, levels=100, cmap='viridis')
fig.colorbar(cf2, ax=axes[1])
axes[1].set_title('Fortran-Trained PINN')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')

# Difference
cf3 = axes[2].contourf(X, T, abs_diff, levels=50, cmap='Reds')
fig.colorbar(cf3, ax=axes[2])
axes[2].set_title('Absolute Difference')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')

plt.tight_layout()
plt.savefig("comparison_plot.png")
print("Comparison plot saved to comparison_plot.png")
plt.show()
