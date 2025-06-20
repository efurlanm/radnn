import tensorflow as tf
import numpy as np
import scipy.io
import time

np.random.seed(1234)
tf.random.set_seed(1234)

def initialize_neural_network(layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(num_layers - 1):
        weight = xavier_initializer(size=[layers[l], layers[l + 1]])
        bias = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(weight)
        biases.append(bias)
    return weights, biases

def xavier_initializer(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

def neural_network_model(X, weights, biases, lower_bound, upper_bound):
    num_layers = len(weights) + 1
    H = 2.0 * (X - lower_bound) / (upper_bound - lower_bound) - 1.0
    for l in range(num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def predict_u(x, t, weights, biases, lower_bound, upper_bound):
    u = neural_network_model(tf.concat([x, t], 1), weights, biases, lower_bound, upper_bound)
    return u

def predict_f(x, t, weights, biases, lambda_1, lambda_2, lower_bound, upper_bound):
    lambda_1_val = lambda_1
    lambda_2_val = tf.exp(lambda_2)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        u = predict_u(x, t, weights, biases, lower_bound, upper_bound)
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
    u_xx = tape.gradient(u_x, x)
    del tape
    f = u_t + lambda_1_val * u * u_x - lambda_2_val * u_xx
    return f

def loss_function(x_input, t_input, u_real, weights, biases, lambda_1, lambda_2, lower_bound, upper_bound):
    u_predicted = predict_u(x_input, t_input, weights, biases, lower_bound, upper_bound)
    f_predicted = predict_f(x_input, t_input, weights, biases, lambda_1, lambda_2, lower_bound, upper_bound)
    loss = tf.reduce_mean(tf.square(u_real - u_predicted)) + tf.reduce_mean(tf.square(f_predicted))
    return loss

@tf.function
def train_step(optimizer, x_input, t_input, u_real, weights, biases, lambda_1, lambda_2, lower_bound, upper_bound):
    with tf.GradientTape() as tape:
        loss = loss_function(x_input, t_input, u_real, weights, biases, lambda_1, lambda_2, lower_bound, upper_bound)
    trainable_variables = weights + biases + [lambda_1, lambda_2]
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss

# --- Main Execution ---
nu = 0.01 / np.pi

# Config
num_train_samples = 2000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
iterations = 10000

# Load data
data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
exact_solution = np.real(data['usol']).T
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = exact_solution.flatten()[:, None]

# Domain bounds
lower_bound = X_star.min(0).astype(np.float32)
upper_bound = X_star.max(0).astype(np.float32)

# Prepare training data
idx = np.random.choice(X_star.shape[0], num_train_samples, replace=False)
X_u_train = X_star[idx, :]
u_train = u_star[idx, :]

# Initialize variables
weights, biases = initialize_neural_network(layers)
lambda_1 = tf.Variable([0.0], dtype=tf.float32)
lambda_2 = tf.Variable([-6.0], dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

x_input = tf.constant(X_u_train[:, 0:1], dtype=tf.float32)
t_input = tf.constant(X_u_train[:, 1:2], dtype=tf.float32)
u_real = tf.constant(u_train, dtype=tf.float32)

# --- Training ---
start_time = time.time()
for it in range(iterations):
    loss_value = train_step(optimizer, x_input, t_input, u_real, weights, biases, lambda_1, lambda_2, lower_bound, upper_bound)
    if it % 100 == 0:
        elapsed = time.time() - start_time
        lambda_1_val = lambda_1.numpy()[0]
        lambda_2_val = np.exp(lambda_2.numpy()[0])
        print(f'Iteration: {it}, Loss: {loss_value:.3e}, Lambda_1: {lambda_1_val:.3f}, '
              f'Lambda_2: {lambda_2_val:.6f}, Time: {elapsed:.2f}')
        start_time = time.time()

x_star_tf = tf.constant(X_star[:, 0:1], dtype=tf.float32)
t_star_tf = tf.constant(X_star[:, 1:2], dtype=tf.float32)
u_pred, _ = predict_u(x_star_tf, t_star_tf, weights, biases, lower_bound, upper_bound), None # Simplified from original predict

lambda_1_val = lambda_1.numpy()[0]
lambda_2_val = np.exp(lambda_2.numpy()[0])
error_lambda_1 = np.abs(lambda_1_val - 1.0) * 100
error_lambda_2 = np.abs(lambda_2_val - nu) / nu * 100

print(f'Error l1: {error_lambda_1:.5f}%')
print(f'Error l2: {error_lambda_2:.5f}%')

# Save results
np.savez('burgers_results_simplified.npz',
         t=t, x=x, exact_solution=exact_solution,
         X_star=X_star, u_star=u_star,
         u_pred=u_pred,
         lambda_1=lambda_1_val,
         lambda_2=lambda_2_val)

print("\nResults saved to burgers_results_simplified.npz")