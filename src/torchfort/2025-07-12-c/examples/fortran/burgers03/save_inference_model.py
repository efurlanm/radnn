
import torch
import torch.nn as nn

# Define the InferenceNet class exactly as in generate_burgers_model.py
# This is necessary for torch.jit.load to know the model's architecture.
class InferenceNet(nn.Module):
    def __init__(self):
        super(InferenceNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

# Define the LossPINN class exactly as in generate_burgers_model.py
class LossPINN(nn.Module):
    def __init__(self, nu: float, inference_net: InferenceNet):
        super(LossPINN, self).__init__()
        self.nu = nu
        self.inference_net = inference_net

    def forward(self, X_f, x0_t0, u0, xb_left_tb, xb_right_tb):
        # The forward pass is not needed for saving, but defining the class
        # structure is important for loading the scripted model.
        pass

# --- Main script ---
print("Loading the trained loss model (burgers_model_trained.pt)...")

# Load the entire trained model from Fortran
# Note: We need to instantiate the class with dummy values to load the state dict.
# The actual values of nu and the untrained inference_net don't matter here.
trained_loss_model = torch.jit.load("burgers_model_trained.pt")

print("Extracting the trained inference_net submodule...")
# The trained model is an instance of LossPINN, and we can access its
# 'inference_net' attribute which holds the trained weights.
# In PyTorch's JIT-scripted models, submodules are attributes.
trained_inference_net = trained_loss_model.inference_net

print("Saving the extracted inference model to burgers_inference_trained.pt...")
# Save the extracted, trained inference model.
# We need to script it before saving.
scripted_trained_inference_net = torch.jit.script(trained_inference_net)
scripted_trained_inference_net.save("burgers_inference_trained.pt")

print("Successfully saved the trained inference model.")
