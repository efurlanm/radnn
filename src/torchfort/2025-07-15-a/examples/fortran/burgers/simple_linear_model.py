import torch
import torch.nn as nn
import torch.jit

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(2, 1) # Input features: 2, Output features: 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

if __name__ == "__main__":
    # Create an instance of the model
    model = SimpleLinearModel()

    # Save the TorchScript model
    scripted_model = torch.jit.script(model)
    scripted_model.save("simple_linear_model.pt")
    print("simple_linear_model.pt saved.")

    # Perform a test inference
    # Input tensor should be (N, features), where N is batch size
    test_input = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    output = scripted_model(test_input)
    print(f"Test input: {test_input}")
    print(f"Test output: {output}")