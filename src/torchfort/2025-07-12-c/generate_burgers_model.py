
import torch
import torch.nn as nn
import numpy as np

# Set the default tensor type to float32 for compatibility with numerical operations
torch.set_default_dtype(torch.float32)

# Define physical parameters
nu = 0.01 / np.pi

class PINN(nn.Module):
    """Simple Feed-Forward Neural Network."""
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

class BurgersPINN(nn.Module):
    """PINN for Burgers' equation, encapsulating the network and the loss logic."""
    def __init__(self, layers):
        super().__init__()
        self.inference_net = PINN(layers)

    def pde_residual(self, x_f):
        """Computes the residual of the Burgers' equation."""
        u = self.inference_net(x_f)

        # Compute first derivatives (u_x, u_t) using a single autograd call for efficiency.
        # torch.autograd.grad returns a tuple, so we extract the first element.
        grads = torch.autograd.grad([u], [x_f], grad_outputs=[torch.ones_like(u)], create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        # Compute the second derivative (u_xx).
        # This is the derivative of u_x with respect to x.
        u_xx_grads = torch.autograd.grad([u_x], [x_f], grad_outputs=[torch.ones_like(u_x)], create_graph=True)[0]
        u_xx = u_xx_grads[:, 0:1]
        
        f = u_t + u * u_x - nu * u_xx
        return f

    def forward(self, x_f, x_0, t_0, u_0, xb_left, tb, xb_right):
        """
        This forward method computes the total loss for training.
        It's designed to be called from Fortran via TorchFort.
        The arguments are the different sets of training points.
        """
        # Loss for PDE residual at collocation points
        f = self.pde_residual(x_f)
        loss_f = torch.mean(f**2)
        
        # Loss for initial condition
        u_0_pred = self.inference_net(torch.cat([x_0, t_0], dim=1))
        loss_0 = torch.mean((u_0_pred - u_0)**2)
        
        # Loss for boundary conditions
        u_b_left_pred = self.inference_net(torch.cat([xb_left, tb], dim=1))
        u_b_right_pred = self.inference_net(torch.cat([xb_right, tb], dim=1))
        # Boundary values are expected to be 0
        loss_b = torch.mean(u_b_left_pred**2) + torch.mean(u_b_right_pred**2)
        
        total_loss = loss_f + loss_0 + loss_b
        return total_loss

class IdentityLoss(nn.Module):
    """
    A simple identity function for the loss.
    TorchFort requires a separate loss function module.
    Since our main model already computes the final loss scalar,
    this loss module just returns that value.
    """
    def __init__(self):
        super().__init__()

    def forward(self, loss):
        return loss

def main():
    """Main function to create and save the models."""
    # Define network architecture
    layers = [2, 10, 10, 10, 10, 10, 10, 10, 10, 1]
    
    # Create model instances
    burgers_model = BurgersPINN(layers)
    loss_fn = IdentityLoss()

    # Use torch.jit.script to compile the models
    scripted_burgers_model = torch.jit.script(burgers_model)
    scripted_loss_fn = torch.jit.script(loss_fn)

    # Save the scripted models
    model_filename = "burgers_model.pt"
    loss_filename = "burgers_loss.pt"
    scripted_burgers_model.save(model_filename)
    scripted_loss_fn.save(loss_filename)
    
    print(f"Model saved to {model_filename}")
    print(f"Loss function saved to {loss_filename}")

if __name__ == '__main__':
    main()
