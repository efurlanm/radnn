import torch
import torch.nn as nn

# Define the PINN class exactly as in burgers1d.py
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net_layers = []
        for i in range(len(layers) - 2):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            net_layers.append(nn.Tanh())
        net_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.net(x)

# Define the BurgersPINN class (from generate_torchscript_model.py) to load the trained model
class BurgersPINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.inference_net = PINN(layers)
        self.register_buffer('nu', torch.tensor(0.01 / torch.pi))

    def forward(self, x_f, x0_cat, xb_left_cat, xb_right_cat):
        # This forward method is for training, but we need it to load the model
        # We will extract inference_net from this.
        return self.inference_net(x_f) # Dummy return

# Define the layers based on burgers1d.py
layers = [2, 10, 10, 10, 10, 10, 10, 10, 10, 1]

# Load the full trained model (which is a BurgersPINN instance)
# This model was saved by burgers_train.f90
model_path = "burgers_model_trained.pt"

# Instantiate BurgersPINN with the correct layers
trained_model_full = torch.jit.load(model_path)

# Extract the inference_net (which is the PINN part)
inference_pinn = trained_model_full.inference_net

# Script and save the extracted inference model
inference_net_filename = "burgers_inference_trained.pt"
torch.jit.script(inference_pinn).save(inference_net_filename)
print(f"Extracted inference network saved to {inference_net_filename}")
