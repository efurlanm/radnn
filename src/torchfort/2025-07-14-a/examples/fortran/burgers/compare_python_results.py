import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the original Python inference results from burgers1d.py
with open("burgers1d_python_original_results.bin", 'rb') as f:
    N_x_orig, N_t_orig = np.fromfile(f, dtype=np.int32, count=2)
    x_orig = np.fromfile(f, dtype=np.float32, count=N_x_orig)
    t_orig = np.fromfile(f, dtype=np.float32, count=N_t_orig)
    u_pred_orig = np.fromfile(f, dtype=np.float32).reshape(N_t_orig, N_x_orig)

# Load the Python inference test results from inference_test.py
with open("burgers1d_python_inference_test_results.bin", 'rb') as f:
    N_x_test, N_t_test = np.fromfile(f, dtype=np.int32, count=2)
    x_test = np.fromfile(f, dtype=np.float32, count=N_x_test)
    t_test = np.fromfile(f, dtype=np.float32, count=N_t_test)
    u_pred_test_flat = np.fromfile(f, dtype=np.float32)
    u_pred_test = u_pred_test_flat.reshape(N_t_test, N_x_test)

# Compare the results
abs_diff = np.abs(u_pred_orig - u_pred_test)
max_abs_diff = np.max(abs_diff)
mean_abs_diff = np.mean(abs_diff)

print(f"Maximum absolute difference (Original vs. Inference Test): {max_abs_diff:.2e}")
print(f"Mean absolute difference (Original vs. Inference Test): {mean_abs_diff:.2e}")

# Plot the results for comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original Python results
X, T = np.meshgrid(x_orig, t_orig)
cf1 = axes[0].contourf(X, T, u_pred_orig, levels=100, cmap='viridis')
fig.colorbar(cf1, ax=axes[0], label='u(x,t)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].set_title('Original Python PINN')

# Python Inference Test results
X_test, T_test = np.meshgrid(x_test, t_test)
cf2 = axes[1].contourf(X_test, T_test, u_pred_test, levels=100, cmap='viridis')
fig.colorbar(cf2, ax=axes[1], label='u(x,t)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
axes[1].set_title('Python Inference Test')

# Absolute difference
cf3 = axes[2].contourf(X, T, abs_diff, levels=100, cmap='Reds')
fig.colorbar(cf3, ax=axes[2], label='Absolute Difference')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
axes[2].set_title('Absolute Difference (Original vs. Test)')

plt.tight_layout()
plt.savefig("comparison_python_inference_test.png")
print("Comparison plot saved to comparison_python_inference_test.png")
# plt.show()
