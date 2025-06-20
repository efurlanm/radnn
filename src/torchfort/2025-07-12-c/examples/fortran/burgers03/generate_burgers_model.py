import torch
import torch.nn as nn
import numpy as np

# This script defines the custom PINN model and an identity loss function,
# then exports them as TorchScript files for the Fortran application to use.

# Define the neural network architecture
class InferenceNet(nn.Module):
    def __init__(self, layers):
        super(InferenceNet, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        # Initialize weights
        for i in range(len(self.layers)):
            nn.init.xavier_normal_(self.layers[i].weight.data, gain=1.0)
            nn.init.zeros_(self.layers[i].bias.data)

    def forward(self, x):
        # Ensure input is of shape [N, 2]
        if len(x.shape) != 2 or x.shape[1] != 2:
            raise ValueError(f"Expected input shape [N, 2], but got {x.shape}")
        
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.activation(layer(x))
            else:
                x = layer(x)
        return x

def main():
    # Parameters
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    # Instantiate models
    inference_net = InferenceNet(layers)

    # Script the models
    scripted_inference_net = torch.jit.script(inference_net)

    # Save the scripted models
    scripted_inference_net.save("burgers_inference_net.pt")

    print("Successfully scripted and saved burgers_inference_net.pt")

if __name__ == "__main__":
    main()
