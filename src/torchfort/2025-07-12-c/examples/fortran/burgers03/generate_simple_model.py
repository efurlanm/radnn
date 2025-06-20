import torch
import torch.nn as nn

# Define a very simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

# --- Main script ---
# Create an instance of the simple network
net = SimpleNet()

# Use torch.jit.script to compile the model
scripted_net = torch.jit.script(net)

# Save the scripted model
scripted_net.save("simple_model.pt")

print("Successfully generated simple_model.pt")
