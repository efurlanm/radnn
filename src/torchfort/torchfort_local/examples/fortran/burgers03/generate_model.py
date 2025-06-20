import torch
import torch.nn as nn
import numpy as np

class PINN_Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        sequential_layers = []
        for i in range(len(layers) - 2):
            sequential_layers.append(nn.Linear(layers[i], layers[i+1]))
            sequential_layers.append(nn.Tanh())
        sequential_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.network = nn.Sequential(*sequential_layers)

    def forward(self, x_t):
        return self.network(x_t)

class BurgersPINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.pinn_net = PINN_Net(layers)
        self.nu = 0.01 / np.pi

    def forward(self, X_f: torch.Tensor, X_0: torch.Tensor, u_0: torch.Tensor, X_b_left: torch.Tensor, X_b_right: torch.Tensor) -> torch.Tensor:
        X_f_grad = X_f.clone().detach().requires_grad_(True)
        
        x = X_f_grad[:, 0:1]
        t = X_f_grad[:, 1:2]

        u = self.pinn_net(torch.cat([x, t], dim=1))

        u_t_list = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)
        u_t = u_t_list[0]
        u_x_list = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        u_x = u_x_list[0]
        u_xx_list = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)
        u_xx = u_xx_list[0]

        f = u_t + u * u_x - self.nu * u_xx
        loss_f = torch.mean(f**2)

        u_0_pred = self.pinn_net(X_0)
        loss_0 = torch.mean((u_0_pred - u_0)**2)

        u_b_left_pred = self.pinn_net(X_b_left)
        u_b_right_pred = self.pinn_net(X_b_right)
        loss_b = torch.mean(u_b_left_pred**2) + torch.mean(u_b_right_pred**2)

        return loss_f + loss_0 + loss_b

    def infer(self, XT_infer: torch.Tensor) -> torch.Tensor:
        return self.pinn_net(XT_infer)

def main():
    layers = [2, 10, 10, 10, 10, 10, 10, 10, 10, 1]
    burgers_pinn = BurgersPINN(layers)

    # Create dummy inputs for tracing
    X_f = torch.randn(100, 2, requires_grad=True)
    X_0 = torch.randn(10, 2)
    u_0 = torch.randn(10, 1)
    X_b_left = torch.randn(10, 2)
    X_b_right = torch.randn(10, 2)

    traced_model = torch.jit.trace(burgers_pinn, (X_f, X_0, u_0, X_b_left, X_b_right))
    traced_model.save("burgers_model.pt")
    print("Model 'burgers_model.pt' exported successfully.")

if __name__ == "__main__":
    main()