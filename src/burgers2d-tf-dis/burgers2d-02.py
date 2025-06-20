import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Parameters ---
nx = 41
ny = 41
nt = 50 # Number of time steps for data generation and PINN training
nu_real = 0.05 # True viscosity for generating "measured" data

# Domain bounds
x_min, x_max = 0.0, 2.0
y_min, y_max = 0.0, 2.0
t_min, t_max = 0.0, nt * 0.001 # Assuming dt = 0.001 from burgers2d-1.py

# Neural Network Architecture
layers = [3, 20, 20, 20, 20, 2] # Input (x,y,t), 4 hidden layers, Output (u,v)

# --- PINN Setup ---
class PINN_Burgers2D:
    def __init__(self, layers, nu_real):
        self.layers = layers
        self.nu_real = nu_real # For generating data, not for discovery

        # Initialize neural network weights and biases
        self.weights, self.biases = self.initialize_neural_network(layers)

        # Discoverable parameter (kinematic viscosity)
        self.nu_pinn = tf.Variable(0.1, dtype=tf.float32, name="nu_pinn") # Initial guess

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def initialize_neural_network(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            weight = self.xavier_initializer(size=[layers[l], layers[l + 1]])
            bias = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    def xavier_initializer(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_network_model(self, X): # X is [x, y, t]
        num_layers = len(self.weights) + 1
        H = X
        for l in range(num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # Output is [u, v]
        return Y

    def predict_u_v(self, x, y, t):
        # Concatenate inputs for the neural network
        X_input = tf.concat([x, y, t], axis=1)
        uv = self.neural_network_model(X_input)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    def pde_residual(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)

            u, v = self.predict_u_v(x, y, t)

            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_t = tape.gradient(u, t)

            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            v_t = tape.gradient(v, t)

        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)

        del tape

        # Burgers 2D equations:
        # u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) = 0
        # v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) = 0

        f_u = u_t + u * u_x + v * u_y - self.nu_pinn * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y - self.nu_pinn * (v_xx + v_yy)

        return f_u, f_v

    def loss_function(self, x_data, y_data, t_data, u_data, v_data, x_pde, y_pde, t_pde):
        # Data loss
        u_pred_data, v_pred_data = self.predict_u_v(x_data, y_data, t_data)
        loss_data = tf.reduce_mean(tf.square(u_data - u_pred_data)) + \
                    tf.reduce_mean(tf.square(v_data - v_pred_data))

        # PDE loss
        f_u_pred, f_v_pred = self.pde_residual(x_pde, y_pde, t_pde)
        loss_pde = tf.reduce_mean(tf.square(f_u_pred)) + \
                   tf.reduce_mean(tf.square(f_v_pred))
        
        return loss_data + loss_pde

    @tf.function
    def train_step(self, x_data, y_data, t_data, u_data, v_data, x_pde, y_pde, t_pde):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x_data, y_data, t_data, u_data, v_data, x_pde, y_pde, t_pde)
        
        trainable_variables = self.weights + self.biases + [self.nu_pinn]
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

# --- Data Generation (similar to burgers2d-1.py for "measured" data) ---
# This part is kept for generating the "measured" data that the PINN will try to match.
# In a real-world scenario, this would be experimental data.
def forward_simulation_for_data(nx, ny, nt, dx, dy, dt, nu_val, u_initial, v_initial):
    u = tf.identity(u_initial)
    v = tf.identity(v_initial)

    for n in range(nt):
        un = tf.identity(u)
        vn = tf.identity(v)

        u_interior = un[1:-1, 1:-1]
        v_interior = vn[1:-1, 1:-1]

        u_x = (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx)
        u_y = (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)
        
        v_x = (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx)
        v_y = (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)
        
        u_xx = (un[1:-1, 2:] - 2 * u_interior + un[1:-1, :-2]) / dx**2
        u_yy = (un[2:, 1:-1] - 2 * u_interior + un[:-2, 1:-1]) / dy**2

        v_xx = (vn[1:-1, 2:] - 2 * v_interior + vn[1:-1, :-2]) / dx**2
        v_yy = (vn[2:, 1:-1] - 2 * v_interior + vn[:-2, 1:-1]) / dy**2
        
        u_conv = u_interior * u_x + v_interior * u_y
        v_conv = u_interior * v_x + v_interior * v_y
        
        u_diff = nu_val * (u_xx + u_yy)
        v_diff = nu_val * (v_xx + v_yy)

        u_next = u_interior - dt * u_conv + dt * u_diff
        v_next = v_interior - dt * v_conv + dt * v_diff
        
        u = tf.tensor_scatter_nd_update(un, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(u_next, [-1]))
        v = tf.tensor_scatter_nd_update(vn, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(v_next, [-1]))
    
    return u, v

# Grid for data generation
x_np = np.linspace(x_min, x_max, nx)
y_np = np.linspace(y_min, y_max, ny)
X_np, Y_np = np.meshgrid(x_np, y_np)
dx = x_np[1] - x_np[0]
dy = y_np[1] - y_np[0]
dt = 0.001 # Consistent with burgers2d-1.py

# Initial conditions for data generation
center_x, center_y = 1.0, 1.0
sigma_x, sigma_y = 0.25, 0.25
u_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
v_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))

u_initial_tf = tf.constant(u_initial_np, dtype=tf.float32)
v_initial_tf = tf.constant(v_initial_np, dtype=tf.float32)

# Generate "measured" data at the final time step
u_medido, v_medido = forward_simulation_for_data(nx, ny, nt, dx, dy, dt, tf.constant(nu_real, dtype=tf.float32), u_initial_tf, v_initial_tf)

# Prepare data for PINN training
# Data points for data loss (from the "measured" solution)
X_data_flat = X_np.flatten()[:, None]
Y_data_flat = Y_np.flatten()[:, None]
T_data_flat = np.full_like(X_data_flat, t_max) # All data at final time
U_data_flat = u_medido.numpy().flatten()[:, None]
V_data_flat = v_medido.numpy().flatten()[:, None]

# Collocation points for PDE loss (randomly sampled across space and time)
num_pde_points = 10000
x_pde = tf.constant(np.random.uniform(x_min, x_max, (num_pde_points, 1)), dtype=tf.float32)
y_pde = tf.constant(np.random.uniform(y_min, y_max, (num_pde_points, 1)), dtype=tf.float32)
t_pde = tf.constant(np.random.uniform(t_min, t_max, (num_pde_points, 1)), dtype=tf.float32)

# Convert data to TensorFlow constants
x_data_tf = tf.constant(X_data_flat, dtype=tf.float32)
y_data_tf = tf.constant(Y_data_flat, dtype=tf.float32)
t_data_tf = tf.constant(T_data_flat, dtype=tf.float32)
u_data_tf = tf.constant(U_data_flat, dtype=tf.float32)
v_data_tf = tf.constant(V_data_flat, dtype=tf.float32)

# --- Training Loop ---
pinn = PINN_Burgers2D(layers, nu_real)
epochs = 5000

print(f"True nu: {nu_real}")
print(f"Initial nu_pinn guess: {pinn.nu_pinn.numpy():.6f}")

for epoch in range(epochs):
    loss_value = pinn.train_step(x_data_tf, y_data_tf, t_data_tf, u_data_tf, v_data_tf, x_pde, y_pde, t_pde)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value.numpy():.6f}, Discovered nu = {pinn.nu_pinn.numpy():.6f}")

print("-" * 50)
print(f"Final Discovered nu: {pinn.nu_pinn.numpy():.6f}")
print(f"True nu: {nu_real}")

# --- Visualization (similar to burgers2d-1.py) ---
# Predict u and v on the original grid at the final time for visualization
X_plot_flat = X_np.flatten()[:, None]
Y_plot_flat = Y_np.flatten()[:, None]
T_plot_flat = np.full_like(X_plot_flat, t_max)

u_pinn_pred_flat, v_pinn_pred_flat = pinn.predict_u_v(
    tf.constant(X_plot_flat, dtype=tf.float32),
    tf.constant(Y_plot_flat, dtype=tf.float32),
    tf.constant(T_plot_flat, dtype=tf.float32)
)

u_pinn_pred = tf.reshape(u_pinn_pred_flat, (ny, nx)).numpy()
v_pinn_pred = tf.reshape(v_pinn_pred_flat, (ny, nx)).numpy()

fig = plt.figure(figsize=(11, 7), dpi=100)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.set_title("Solução Medida (nu real)")
surf1 = ax1.plot_surface(X_np, Y_np, u_medido.numpy(), cmap=cm.viridis)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('u (velocidade em x)')

ax2.set_title(f"Solução PINN (nu = {pinn.nu_pinn.numpy():.4f})")
surf2 = ax2.plot_surface(X_np, Y_np, u_pinn_pred, cmap=cm.viridis)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('u (velocidade em x)')

plt.tight_layout()
plt.savefig('burgers2d_pinn_results.pdf')
