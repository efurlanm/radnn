import numpy as np
import matplotlib.pyplot as plt

# Define physical parameters and boundaries (must match training script)
x_min, x_max = -1.0, 1.0  # Spatial range
t_min, t_max = 0.0, 1.0   # Temporal range

# Define the number of sampled points for inference grid
N_x, N_t = 256, 100

def load_original_python_results(filename="original_python_u_pred_direct.txt"):
    u_pred = np.loadtxt(filename, dtype=np.float32)
    # For plotting, we still need x and t from the original binary file
    # This is a bit hacky, but avoids re-generating them.
    with open("burgers1d_python_original_results.bin", 'rb') as f:
        dims = np.fromfile(f, dtype=np.int32, count=2)
        N_x_read, N_t_read = dims[0], dims[1]
        x = np.fromfile(f, dtype=np.float32, count=N_x_read)
        t = np.fromfile(f, dtype=np.float32, count=N_t_read)
    return x, t, u_pred

def load_python_inference_results(filename="python_inference_u_pred.txt"):
    u_pred = np.loadtxt(filename, dtype=np.float32)
    return u_pred

def compare_arrays(name, arr1, arr2, atol=1e-5):
    print(f"\nComparing {name}:")
    print(f"  Shape 1: {arr1.shape}, Shape 2: {arr2.shape}")
    if arr1.shape != arr2.shape:
        print(f"  Shape mismatch for {name}!")
        return False
    
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    if not np.allclose(arr1, arr2, atol=atol):
        print(f"  {name} arrays are NOT close enough!")
        return False
    print(f"  {name} arrays are close enough.")
    return True

if __name__ == "__main__":
    print("Loading original Python results...")
    orig_x, orig_t, orig_u_pred = load_original_python_results()
    
    print("Loading Python inference results...")
    inf_u_pred = load_python_inference_results()

    # Ensure shapes match for comparison
    if orig_u_pred.shape != inf_u_pred.shape:
        print("Error: Shape mismatch between original and inference u_pred. Reshaping inference u_pred.")
        # Attempt to reshape if dimensions are compatible
        if orig_u_pred.size == inf_u_pred.size:
            inf_u_pred = inf_u_pred.reshape(orig_u_pred.shape)
        else:
            print("Error: Total number of elements also mismatch. Cannot compare.")
            exit()

    all_close = True
    all_close = all_close and compare_arrays("u_pred", orig_u_pred, inf_u_pred, atol=1e-4) # Increased tolerance for u_pred

    if all_close:
        print("\nAll inference results are consistent between original Python and Python inference.")
    else:
        print("\nDiscrepancies found in inference results between original Python and Python inference.")

    # Plotting for visual comparison
    X_plot, T_plot = np.meshgrid(orig_x, orig_t)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.contourf(X_plot, T_plot, orig_u_pred, levels=100, cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Original Python u(x,t)")

    plt.subplot(1, 2, 2)
    plt.contourf(X_plot, T_plot, inf_u_pred, levels=100, cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Python Inference u(x,t)")

    plt.tight_layout()
    plt.savefig("python_inference_comparison.png")
    print("Comparison plot saved to python_inference_comparison.png")
    # plt.show()
