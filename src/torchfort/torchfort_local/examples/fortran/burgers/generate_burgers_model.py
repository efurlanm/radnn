import torch
from typing import List, Optional
import numpy as np

# Fixed random seed for reproducibility
SEED = 42

class BurgersPINN(torch.nn.Module):
    def __init__(self):
        super(BurgersPINN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 1)
        )
        self.nu = 0.01 / torch.pi

    def forward(self, t_collocation: torch.Tensor, x_collocation: torch.Tensor,
                t_initial: torch.Tensor, x_initial: torch.Tensor, u_initial_true: torch.Tensor,
                t_boundary: torch.Tensor, x_boundary: torch.Tensor, u_boundary_true: torch.Tensor) -> torch.Tensor:

        # PDE loss
        t_collocation_grad = t_collocation.clone().detach().requires_grad_(True)
        x_collocation_grad = x_collocation.clone().detach().requires_grad_(True)
        u_collocation = self.net(torch.cat([t_collocation_grad, x_collocation_grad], dim=1))

        ones_like_u = torch.ones_like(u_collocation)
        grad_outputs_u: List[Optional[torch.Tensor]] = [ones_like_u]

        u_t_opt = torch.autograd.grad([u_collocation], [t_collocation_grad], grad_outputs=grad_outputs_u, create_graph=True)[0]
        u_t = u_t_opt if u_t_opt is not None else torch.zeros_like(u_collocation) # Handle Optional[Tensor]

        u_x_opt = torch.autograd.grad([u_collocation], [x_collocation_grad], grad_outputs=grad_outputs_u, create_graph=True)[0]
        u_x = u_x_opt if u_x_opt is not None else torch.zeros_like(u_collocation) # Handle Optional[Tensor]

        ones_like_ux = torch.ones_like(u_x)
        grad_outputs_ux: List[Optional[torch.Tensor]] = [ones_like_ux]
        u_xx_opt = torch.autograd.grad([u_x], [x_collocation_grad], grad_outputs=grad_outputs_ux, create_graph=True)[0]
        u_xx = u_xx_opt if u_xx_opt is not None else torch.zeros_like(u_x) # Handle Optional[Tensor]

        f = u_t + u_collocation * u_x - self.nu * u_xx
        loss_f = torch.mean(f**2)

        # Initial condition loss
        u_initial_pred = self.net(torch.cat([t_initial, x_initial], dim=1))
        loss_i = torch.mean((u_initial_pred - u_initial_true)**2)

        # Boundary condition loss
        u_boundary_pred = self.net(torch.cat([t_boundary, x_boundary], dim=1))
        loss_b = torch.mean((u_boundary_pred - u_boundary_true)**2)

        return loss_f + loss_i + loss_b

class IdentityLoss(torch.nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()

    def forward(self, model_output: torch.Tensor, dummy_label: torch.Tensor) -> torch.Tensor:
        return model_output

class SimpleInferenceNet(torch.nn.Module):
    def __init__(self, net_module: torch.nn.Sequential):
        super(SimpleInferenceNet, self).__init__()
        self.net = net_module

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, x], dim=1))

def generate_initial_models():
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    burgers_pinn = BurgersPINN().to(device)
    print("BurgersPINN model (initial):", burgers_pinn)

    identity_loss = IdentityLoss().to(device)
    print("Identity Loss module (initial):", identity_loss)

    # Script and save the training model (initial state)
    model_jit = torch.jit.script(burgers_pinn)
    model_jit.save("../examples/fortran/burgers/burgers_model.pt")

    # Script and save the loss model
    loss_jit = torch.jit.script(identity_loss)
    loss_jit.save("../examples/fortran/burgers/burgers_loss.pt")

    # Script and save the simple inference network (initial state)
    simple_inference_net = SimpleInferenceNet(burgers_pinn.net).to(device)
    inference_net_jit = torch.jit.script(simple_inference_net)
    inference_net_jit.save("../examples/fortran/burgers/burgers_inference_net.pt")

if __name__ == "__main__":
    generate_initial_models()