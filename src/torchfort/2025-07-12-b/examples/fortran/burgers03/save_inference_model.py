import torch
import torch.nn as nn

# Define the PINN class exactly as in generate_burgers_model.py
# This is necessary for torch.jit.load to know the model's architecture.
class BurgersPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc_out = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        x = self.activation(self.fc7(x))
        x = self.activation(self.fc8(x))
        x = self.fc_out(x)
        return x

# Define the wrapper class to match the saved model structure
class BurgersModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.pinn = BurgersPINN()
        # nu is not needed for loading the state dict
        self.nu = 0.0 

    def forward(self, X_f, x0_t0, xb_left_tb, xb_right_tb):
        # This forward pass is not used, but the structure is needed for loading
        pass

# --- Main script ---
print("Loading the trained model from Fortran (burgers_model_trained.pt)...")

# Load the entire trained model saved by Fortran
trained_model_wrapper = torch.jit.load("burgers_model_trained.pt")

print("Extracting the trained PINN submodule...")
# The trained model is an instance of BurgersModelWrapper, and we can access its
# 'pinn' attribute which holds the trained weights.
# In PyTorch's JIT-scripted models, submodules are attributes.
trained_pinn = trained_model_wrapper.pinn

print("Saving the extracted inference model to burgers_inference_trained.pt...")
# Save the extracted, trained inference model.
# We need to script it again before saving.
scripted_trained_pinn = torch.jit.script(trained_pinn)
scripted_trained_pinn.save("burgers_inference_trained.pt")

print("Successfully saved the trained inference model.")