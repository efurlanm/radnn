import numpy as np

# Load Fortran XT_tensor
with open("fortran_xt_tensor.bin", 'rb') as f:
    fortran_xt_flat = np.fromfile(f, dtype=np.float32)
    # Assuming XT_tensor was saved as (2, total_points) in Fortran
    # and total_points = N_x * N_t = 256 * 100 = 25600
    # So, total elements = 2 * 25600 = 51200
    # Reshape to (total_points, 2) to match Python's (N, features)
    fortran_xt = fortran_xt_flat.reshape(-1, 2)

# Load Python XT_tensor
with open("python_xt_tensor.bin", 'rb') as f:
    python_xt_flat = np.fromfile(f, dtype=np.float32)
    python_xt = python_xt_flat.reshape(-1, 2)

# Compare
abs_diff = np.abs(fortran_xt - python_xt)
max_abs_diff = np.max(abs_diff)
mean_abs_diff = np.mean(abs_diff)

print(f"Max absolute difference in XT_tensor: {max_abs_diff:.2e}")
print(f"Mean absolute difference in XT_tensor: {mean_abs_diff:.2e}")

# Optionally, print some values to inspect
print("\nFirst 5 rows of Fortran XT_tensor:")
print(fortran_xt[:5])
print("\nFirst 5 rows of Python XT_tensor:")
print(python_xt[:5])

print("\nLast 5 rows of Fortran XT_tensor:")
print(fortran_xt[-5:])
print("\nLast 5 rows of Python XT_tensor:")
print(python_xt[-5:])
