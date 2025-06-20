import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net_layers = []
        for i in range(len(layers) - 2):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            net_layers.append(nn.Tanh())
        net_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net_layers)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)

def main():
    layers = [2, 10, 10, 10, 10, 10, 10, 10, 10, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create and save the PINN model
    pinn_model = PINN(layers).to(device)
    scripted_pinn_model = torch.jit.script(pinn_model)
    scripted_pinn_model.save("test_pinn_model.pt")
    print("PINN model saved to test_pinn_model.pt")

    # Load the saved model
    loaded_model = torch.jit.load("test_pinn_model.pt").to(device)
    loaded_model.eval()
    print("PINN model loaded from test_pinn_model.pt")

    # Create a dummy input tensor
    dummy_input = torch.randn(10, 2, dtype=torch.float32).to(device)

    # Perform inference
    with torch.no_grad():
        output = loaded_model(dummy_input)
    print(f"Inference successful. Output shape: {output.shape}")

if __name__ == '__main__':
    main()
