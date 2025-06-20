import torch
import numpy as np

# Define physical parameters and boundaries (matching burgers1d.py)
x_min, x_max = -1.0, 1.0  # Spatial range
t_min, t_max = 0.0, 1.0   # Temporal range

# Generate a grid for visualization (matching burgers1d.py)
N_x, N_t = 256, 100
x = np.linspace(x_min, x_max, N_x)
t = np.linspace(t_min, t_max, N_t)
X, T = np.meshgrid(x, t)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to PyTorch tensor and move to device
XT_tensor = torch.tensor(XT, dtype=torch.float32).to(device)

# Load the TorchScript model (relative path as script will be run from this directory)
model_path = "burgers_inference_trained.pt"
print(f"Loading model from: {model_path}")
model = torch.jit.load(model_path)
model.eval() # Set model to evaluation mode

print("Model loaded successfully.")

# Perform inference
with torch.no_grad():
    u_pred_tensor = model(XT_tensor)

# Move prediction back to CPU and convert to NumPy
u_pred = u_pred_tensor.cpu().numpy()

print(f"Inference successful. Predicted output shape: {u_pred.shape}")
print(f"First 10 predicted values: {u_pred.flatten()[:10]}")
print(f"Last 10 predicted values: {u_pred.flatten()[-10:]}")

# Save results to a binary file for comparison
output_filename_test = "burgers1d_python_inference_test_results.bin"
with open(output_filename_test, 'wb') as f:
    f.write(np.array([N_x, N_t], dtype=np.int32).tobytes())
    f.write(x.astype(np.float32).tobytes())
    f.write(t.astype(np.float32).tobytes())
    f.write(u_pred.astype(np.float32).tobytes())
print(f"Inference test results saved to {output_filename_test}")
