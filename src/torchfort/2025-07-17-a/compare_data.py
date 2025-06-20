import numpy as np

def load_python_data():
    X_f = np.loadtxt("X_f.txt", dtype=np.float32)
    x0_t0 = np.loadtxt("x0_t0.txt", dtype=np.float32)
    u0 = np.loadtxt("u0.txt", dtype=np.float32)
    xb_left_tb = np.loadtxt("xb_left_tb.txt", dtype=np.float32)
    xb_right_tb = np.loadtxt("xb_right_tb.txt", dtype=np.float32)

    x0 = x0_t0[:, 0:1]
    t0 = x0_t0[:, 1:2]
    xb_left = xb_left_tb[:, 0:1]
    tb = xb_left_tb[:, 1:2]
    xb_right = xb_right_tb[:, 0:1]
    ub_left = np.zeros_like(tb)
    ub_right = np.zeros_like(tb)

    return X_f, x0, t0, u0, xb_left, tb, xb_right, ub_left, ub_right

def load_fortran_data():
    X_f = np.loadtxt("X_f.txt", dtype=np.float32)
    x0_t0 = np.loadtxt("x0_t0.txt", dtype=np.float32)
    u0 = np.loadtxt("u0.txt", dtype=np.float32)
    xb_left_tb = np.loadtxt("xb_left_tb.txt", dtype=np.float32)
    xb_right_tb = np.loadtxt("xb_right_tb.txt", dtype=np.float32)

    x0 = x0_t0[:, 0:1]
    t0 = x0_t0[:, 1:2]
    xb_left = xb_left_tb[:, 0:1]
    tb = xb_left_tb[:, 1:2]
    xb_right = xb_right_tb[:, 0:1]
    ub_left = np.zeros_like(tb)
    ub_right = np.zeros_like(tb)

    return X_f, x0, t0, u0, xb_left, tb, xb_right, ub_left, ub_right

def compare_arrays(name, arr1, arr2):
    print(f"\nComparing {name}:")
    print(f"  Shape Python: {arr1.shape}, Fortran: {arr2.shape}")
    if arr1.shape != arr2.shape:
        print(f"  Shape mismatch for {name}!")
        return False
    
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    if not np.allclose(arr1, arr2, atol=1e-5):
        print(f"  {name} arrays are NOT close enough!")
        return False
    print(f"  {name} arrays are close enough.")
    return True

if __name__ == "__main__":
    print("Loading Python data...")
    py_X_f, py_x0, py_t0, py_u0, py_xb_left, py_tb, py_xb_right, py_ub_left, py_ub_right = load_python_data()
    print("Loading Fortran data...")
    ft_X_f, ft_x0, ft_t0, ft_u0, ft_xb_left, ft_tb, ft_xb_right, ft_ub_left, ft_ub_right = load_fortran_data()

    all_close = True
    all_close = all_close and compare_arrays("X_f", py_X_f, ft_X_f)
    all_close = all_close and compare_arrays("x0", py_x0, ft_x0)
    all_close = all_close and compare_arrays("t0", py_t0, ft_t0)
    all_close = all_close and compare_arrays("u0", py_u0, ft_u0)
    all_close = all_close and compare_arrays("xb_left", py_xb_left, ft_xb_left)
    all_close = all_close and compare_arrays("tb", py_tb, ft_tb)
    all_close = all_close and compare_arrays("xb_right", py_xb_right, ft_xb_right)
    all_close = all_close and compare_arrays("ub_left", py_ub_left, ft_ub_left)
    all_close = all_close and compare_arrays("ub_right", py_ub_right, ft_ub_right)

    if all_close:
        print("\nAll data arrays are consistent between Python and Fortran.")
    else:
        print("\nDiscrepancies found in data arrays between Python and Fortran.")
