# Conceptual Differences between Burgers' Equation Solvers

This document provides a detailed comparison of three Python scripts: `burgers2d-01.py` (no PINN), `burgers2d-02.py` (PINN), and `discovery1d.py` (1D Raissi2019). The analysis focuses on their conceptual approaches to solving and discovering parameters for Burgers' equation, drawing parallels with the Physics-Informed Neural Networks (PINN) framework introduced by Raissi et al. (2019).

## 1. Conceptual Differences between `burgers2d-01.py` and `burgers2d-02.py`

These two scripts address the inverse problem of discovering the kinematic viscosity coefficient (`nu`) for the 2D Burgers' equation. However, they employ fundamentally different methodologies.

### `burgers2d-01.py`: Traditional Numerical Solver with Parameter Optimization

This script represents a more traditional approach to parameter discovery, where a numerical solver (finite differences) is coupled with an optimization routine.

* **Solution Representation**: The solution `u(x,y,t)` and `v(x,y,t)` are represented on a discrete spatial grid and evolved over time using an explicit finite difference scheme. The `forward_simulation` function directly implements the numerical discretization of the PDE.
  
  ```python
  # From burgers2d-01.py
  # Spatial derivatives using finite differences
  u_x = (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx)
  u_yy = (un[2:, 1:-1] - 2 * u_interior + un[:-2, 1:-1]) / dy**2
  
  # PDE update (explicit finite difference scheme)
  u_next = u_interior - dt * u_conv + dt * u_diff
  ```

* **PDE Enforcement**: The Partial Differential Equation (PDE) is inherently satisfied by the chosen finite difference numerical scheme. The accuracy of the solution depends on the discretization parameters (`dx`, `dy`, `dt`) and the stability of the scheme.

* **Parameter Discovery**: The unknown parameter `nu` (named `nu_guess`) is treated as a `tf.Variable`. The script minimizes the Mean Squared Error (MSE) between the output of this finite difference simulation (run with the current `nu_guess`) and a set of "measured" (ground truth) data. TensorFlow's `GradientTape` is used to compute gradients of this MSE loss with respect to `nu_guess`, allowing the optimizer to update `nu_guess`.
  
  ```python
  # From burgers2d-01.py
  nu_guess = tf.Variable(0.1, dtype=tf.float32, name="nu_guess")
  
  # Loss function based on comparison with "measured" data
  loss = tf.reduce_mean(tf.square(u_predicted - u_medido)) + tf.reduce_mean(tf.square(v_predicted - v_medido))
  
  # Gradients are computed through the forward_simulation
  gradients = tape.gradient(loss, nu_guess)
  optimizer.apply_gradients([(gradients, nu_guess)])
  ```

* **Derivative Calculation**: Spatial derivatives within the `forward_simulation` are explicitly approximated using finite difference formulas.

### `burgers2d-02.py`: Physics-Informed Neural Network (PINN)

This script implements a Physics-Informed Neural Network (PINN) approach, aligning with the methodology described in Raissi et al. (2019).

* **Solution Representation**: The solutions `u(x,y,t)` and `v(x,y,t)` are approximated by a deep neural network. The network takes spatial (`x`, `y`) and temporal (`t`) coordinates as input and outputs the corresponding `u` and `v` values. This allows for a continuous representation of the solution across the spatio-temporal domain.
  
  ```python
  # From burgers2d-02.py
  # Neural network input and output
  X_input = tf.concat([x, y, t], axis=1)
  uv = self.neural_network_model(X_input) # Output is [u, v]
  u = uv[:, 0:1]
  v = uv[:, 1:2]
  ```

* **PDE Enforcement**: The PDE is enforced by incorporating its residual directly into the loss function. The total loss comprises two main components:
  
  1. **Data Loss**: Minimizes the difference between the neural network's predictions and the available "measured" data points.
  2. **PDE Loss (Physics-informed term)**: Minimizes the residual of the Burgers' equations when evaluated at a set of "collocation points" (randomly sampled points in the spatio-temporal domain). This term ensures that the neural network's solution adheres to the underlying physics. This concept is central to PINNs (Raissi et al., 2019, Section 3.1, Eq. 4).
  
  ```python
  # From burgers2d-02.py
  # PDE residuals (f_u and f_v should ideally be zero)
  f_u = u_t + u * u_x + v * u_y - self.nu_pinn * (u_xx + u_yy)
  f_v = v_t + u * v_x + v * v_y - self.nu_pinn * (v_xx + v_yy)
  
  # Loss function combining data and PDE residuals
  loss_data = tf.reduce_mean(tf.square(u_data - u_pred_data)) + tf.reduce_mean(tf.square(v_data - v_pred_data))
  loss_pde = tf.reduce_mean(tf.square(f_u)) + tf.reduce_mean(tf.square(f_v))
  total_loss = loss_data + loss_pde
  ```

* **Parameter Discovery**: The unknown parameter `nu` (named `nu_pinn`) is treated as a trainable `tf.Variable` within the PINN model. It is optimized simultaneously with the neural network's weights and biases during the training process, as part of minimizing the combined data and PDE loss.
  
  ```python
  # From burgers2d-02.py
  self.nu_pinn = tf.Variable(0.1, dtype=tf.float32, name="nu_pinn")
  # ...
  trainable_variables = self.weights + self.biases + [self.nu_pinn]
  gradients = tape.gradient(loss, trainable_variables)
  self.optimizer.apply_gradients(zip(gradients, trainable_variables))
  ```

* **Derivative Calculation**: All partial derivatives required for the PDE residuals (e.g., `u_x`, `u_y`, `u_t`, `u_xx`, `u_yy`) are computed using TensorFlow's **Automatic Differentiation** (`tf.GradientTape`). This eliminates the need for manual finite difference approximations and allows for exact gradients through the neural network, a key enabler for PINNs (Raissi et al., 2019, Section 2).
  
  ```python
  # From burgers2d-02.py
  with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      tape.watch(y)
      tape.watch(t)
      u, v = self.predict_u_v(x, y, t)
      u_x = tape.gradient(u, x)
      u_y = tape.gradient(u, y)
      u_t = tape.gradient(u, t)
  u_xx = tape.gradient(u_x, x)
  v_yy = tape.gradient(v_y, y)
  ```

## 2. Conceptual Differences between `discovery1d.py` and `burgers2d-02.py`

Both `discovery1d.py` and `burgers2d-02.py` are implementations of the PINN framework for parameter discovery. The primary differences lie in the dimensionality of the Burgers' equation they solve and the specifics of their implementation for that dimensionality.

### `discovery1d.py`: 1D Burgers' Equation PINN

This script is a direct adaptation of the 1D Burgers' equation example for parameter discovery presented in Raissi et al. (2019, Appendix B.1).

* **Dimensionality**: Solves the 1D Burgers' equation:
  `u_t + λ1 * u * u_x - λ2 * u_xx = 0`
  (Raissi et al., 2019, Eq. B.1).

* **Neural Network Input/Output**: The neural network takes two inputs (`x`, `t`) and outputs a single scalar `u` (the velocity component).
  
  ```python
  # From discovery1d.py
  def predict_u(x, t, weights, biases, lower_bound, upper_bound):
      u = neural_network_model(tf.concat([x, t], 1), weights, biases, lower_bound, upper_bound)
      return u
  ```

* **PDE Residuals**: Computes a single PDE residual `f` corresponding to the 1D Burgers' equation.
  
  ```python
  # From discovery1d.py
  f = u_t + lambda_1_val * u * u_x - lambda_2_val * u_xx
  ```

* **Parameterization of `nu`**: The diffusion coefficient (`nu`) is represented by `lambda_2`, which is parameterized as `tf.exp(lambda_2)` to ensure its positivity during optimization. `lambda_1` represents the convection coefficient.
  
  ```python
  # From discovery1d.py
  lambda_1 = tf.Variable([0.0], dtype=tf.float32)
  lambda_2 = tf.Variable([-6.0], dtype=tf.float32) # nu = exp(lambda_2)
  # ...
  lambda_2_val = np.exp(lambda_2.numpy()[0])
  ```

* **Data**: Uses pre-loaded data from a `.mat` file (`burgers_shock.mat`) for training.

### `burgers2d-02.py`: 2D Burgers' Equation PINN

This script extends the PINN concept to the 2D Burgers' equation.

* **Dimensionality**: Solves the coupled 2D Burgers' equations:
  `u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) = 0`
  `v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) = 0`

* **Neural Network Input/Output**: The neural network takes three inputs (`x`, `y`, `t`) and outputs two scalar values (`u`, `v`), representing the velocity components in the x and y directions, respectively.
  
  ```python
  # From burgers2d-02.py
  def predict_u_v(self, x, y, t):
      X_input = tf.concat([x, y, t], axis=1)
      uv = self.neural_network_model(X_input)
      u = uv[:, 0:1]
      v = uv[:, 1:2]
      return u, v
  ```

* **PDE Residuals**: Computes two coupled PDE residuals, `f_u` and `f_v`, corresponding to the two equations of the 2D Burgers' system.
  
  ```python
  # From burgers2d-02.py
  f_u = u_t + u * u_x + v * u_y - self.nu_pinn * (u_xx + u_yy)
  f_v = v_t + u * v_x + v * v_y - self.nu_pinn * (v_xx + v_yy)
  ```

* **Parameterization of `nu`**: The diffusion coefficient `nu_pinn` is directly optimized as a `tf.Variable`, without an exponential transformation.
  
  ```python
  # From burgers2d-02.py
  self.nu_pinn = tf.Variable(0.1, dtype=tf.float32, name="nu_pinn")
  ```

* **Data**: Generates "measured" data internally using a separate finite difference simulation, which then serves as the data component of the PINN's loss function.

### Commonalities (as PINNs)

Despite their differences in dimensionality and specific equation forms, both `discovery1d.py` and `burgers2d-02.py` fundamentally adhere to the PINN framework as outlined by Raissi et al. (2019):

* **Neural Network as Solution Approximator**: Both use a deep neural network to approximate the solution of the PDE.
* **Automatic Differentiation**: Both leverage automatic differentiation (via `tf.GradientTape`) to compute all necessary derivatives of the neural network output with respect to its inputs (space and time). This is crucial for calculating the PDE residuals.
* **Physics-Informed Loss Function**: Both incorporate the PDE residuals into their loss functions, ensuring that the learned solution not only fits the available data but also satisfies the governing physical laws.
* **Parameter Discovery**: Both treat the unknown parameters of the PDE as trainable variables within the neural network optimization process, allowing for data-driven discovery of these parameters.

This detailed comparison highlights the evolution from a traditional numerical approach to a full PINN implementation, and then the extension of PINNs to higher-dimensional and more complex PDE systems, consistent with the advancements discussed in the Raissi et al. (2019) paper.
