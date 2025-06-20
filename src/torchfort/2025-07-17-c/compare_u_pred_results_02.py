import numpy as np

def load_binary_results(filename):
    """Loads N_x, N_t, x, t, and u_pred from a binary file."""
    with open(filename, 'rb') as f:
        N_x, N_t = np.frombuffer(f.read(8), dtype=np.int32)
        x = np.frombuffer(f.read(N_x * 4), dtype=np.float32)
        t = np.frombuffer(f.read(N_t * 4), dtype=np.float32)
        u_pred = np.frombuffer(f.read(N_x * N_t * 4), dtype=np.float32).reshape(N_t, N_x)
    return N_x, N_t, x, t, u_pred

# Load results from original Python script
python_original_file = "burgers1d_python_original_results.bin"
N_x_py, N_t_py, x_py, t_py, u_pred_py = load_binary_results(python_original_file)
print(f"Loaded Python original results from {python_original_file}")

# Load results from Fortran-trained model (inference performed in Fortran)
fortran_trained_file = "fortran_trained_u_pred_02.bin"
N_x_ft, N_t_ft, x_ft, t_ft, u_pred_ft = load_binary_results(fortran_trained_file)
print(f"Loaded Fortran trained results from {fortran_trained_file}")

# --- Compare N_x and N_t ---
print("\n--- Comparing Grid Dimensions ---")
print(f"N_x (Python): {N_x_py}, N_x (Fortran): {N_x_ft}")
print(f"N_t (Python): {N_t_py}, N_t (Fortran): {N_t_ft}")
if N_x_py == N_x_ft and N_t_py == N_t_ft:
    print("Grid dimensions match.")
else:
    print("Grid dimensions DO NOT match. This is an error.")

# --- Compare x and t arrays ---
print("\n--- Comparing x and t arrays ---")
max_abs_diff_x = np.max(np.abs(x_py - x_ft))
mean_abs_diff_x = np.mean(np.abs(x_py - x_ft))
print(f"x array - Max Absolute Difference: {max_abs_diff_x:.5e}")
print(f"x array - Mean Absolute Difference: {mean_abs_diff_x:.5e}")

max_abs_diff_t = np.max(np.abs(t_py - t_ft))
mean_abs_diff_t = np.mean(np.abs(t_py - t_ft))
print(f"t array - Max Absolute Difference: {max_abs_diff_t:.5e}")
print(f"t array - Mean Absolute Difference: {mean_abs_diff_t:.5e}")

# --- Compare u_pred ---
print("\n--- Comparing u_pred (Predicted Solution) ---")
abs_diff_u_pred = np.abs(u_pred_py - u_pred_ft)
max_abs_diff_u_pred = np.max(abs_diff_u_pred)
mean_abs_diff_u_pred = np.mean(abs_diff_u_pred)

print(f"u_pred - Max Absolute Difference: {max_abs_diff_u_pred:.5e}")
print(f"u_pred - Mean Absolute Difference: {mean_abs_diff_u_pred:.5e}")

# Define a tolerance for numerical comparison
tolerance = 1e-5 # Adjust as needed based on expected precision

if max_abs_diff_u_pred < tolerance:
    print(f"u_pred results are numerically similar within tolerance ({tolerance:.1e}).")
else:
    print(f"u_pred results are NOT numerically similar within tolerance ({tolerance:.1e}).")
    print("Further investigation needed for numerical divergence.")
