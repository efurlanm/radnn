import numpy as np

# Load Fortran inference results
with open("burgers1d_fortran_inference_results.bin", 'rb') as f:
    N_x_fortran, N_t_fortran = np.fromfile(f, dtype=np.int32, count=2)
    x_fortran = np.fromfile(f, dtype=np.float32, count=N_x_fortran)
    t_fortran = np.fromfile(f, dtype=np.float32, count=N_t_fortran)
    u_pred_fortran = np.fromfile(f, dtype=np.float32).reshape(N_t_fortran, N_x_fortran)

# Load Fortran-trained model's Python inference results
with open("fortran_trained_model_python_inference_results.bin", 'rb') as f:
    N_x_python_inf, N_t_python_inf = np.fromfile(f, dtype=np.int32, count=2)
    x_python_inf = np.fromfile(f, dtype=np.float32, count=N_x_python_inf)
    t_python_inf = np.fromfile(f, dtype=np.float32, count=N_t_python_inf)
    u_pred_python_inf = np.fromfile(f, dtype=np.float32).reshape(N_t_python_inf, N_x_python_inf)

# Compare the results
abs_diff = np.abs(u_pred_fortran - u_pred_python_inf)
max_abs_diff = np.max(abs_diff)
mean_abs_diff = np.mean(abs_diff)

print(f"Max absolute difference (Fortran Inference vs. Fortran-trained Python Inference): {max_abs_diff:.2e}")
print(f"Mean absolute difference (Fortran Inference vs. Fortran-trained Python Inference): {mean_abs_diff:.2e}")

# Optionally, print some values to inspect
print("\nFirst 5x5 block of Fortran Inference u_pred:")
print(u_pred_fortran[:5, :5])
print("\nFirst 5x5 block of Fortran-trained Python Inference u_pred:")
print(u_pred_python_inf[:5, :5])
