import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

# Este programa resolve o problema inverso para a equação de Burgers 2D
# O objetivo é descobrir o coeficiente de difusão (nu) a partir de dados medidos.

# --- Parâmetros do Problema ---
nx = 41 # Número de pontos na direção x
ny = 41 # Número de pontos na direção y
nt_medido = 50 # Passos de tempo para a solução "medida"
nt_treino = 50 # Passos de tempo para o treino (deve ser igual a nt_medido)
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
dt = 0.001

# --- Geração de Dados "Medidos" (Solução de Referência) ---
# Usamos um valor conhecido de nu para gerar os dados de "referência" ou "medidos".
nu_real = 0.05
print(f"Valor real de nu para a simulação: {nu_real}")

def forward_simulation(nx, ny, nt, dx, dy, dt, nu_val, u_initial, v_initial):
    """
    Simula a equação de Burgers 2D (método explícito).
    Aceita um tensor para nu_val, permitindo o cálculo de gradientes.
    """
    u = tf.identity(u_initial)
    v = tf.identity(v_initial)

    for n in range(nt):
        un = tf.identity(u)
        vn = tf.identity(v)

        # Extrai os tensores para os pontos internos (sem as bordas)
        u_interior = un[1:-1, 1:-1]
        v_interior = vn[1:-1, 1:-1]

        # Calcula as derivadas espaciais usando diferenças finitas centrais
        # As fatias são ajustadas para garantir que todas tenham a mesma forma (ny-2, nx-2)
        u_x = (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx)
        u_y = (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)
        
        v_x = (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx)
        v_y = (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)
        
        u_xx = (un[1:-1, 2:] - 2 * u_interior + un[1:-1, :-2]) / dx**2
        u_yy = (un[2:, 1:-1] - 2 * u_interior + un[:-2, 1:-1]) / dy**2

        v_xx = (vn[1:-1, 2:] - 2 * v_interior + vn[1:-1, :-2]) / dx**2
        v_yy = (vn[2:, 1:-1] - 2 * v_interior + vn[:-2, 1:-1]) / dy**2
        
        # Termos de convecção (u*u_x + v*u_y)
        u_conv = u_interior * u_x + v_interior * u_y
        v_conv = u_interior * v_x + v_interior * v_y
        
        # Termos de difusão (nu*(u_xx + u_yy))
        u_diff = nu_val * (u_xx + u_yy)
        v_diff = nu_val * (v_xx + v_yy)

        # Atualização explícita
        u_next = u_interior - dt * u_conv + dt * u_diff
        v_next = v_interior - dt * v_conv + dt * v_diff
        
        # Atualiza a grade original com os novos valores internos
        u = tf.tensor_scatter_nd_update(un, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(u_next, [-1]))
        v = tf.tensor_scatter_nd_update(vn, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(v_next, [-1]))
    
    return u, v

# Condições iniciais (as mesmas do programa anterior)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
center_x, center_y = 1.0, 1.0
sigma_x, sigma_y = 0.25, 0.25
u_initial_np = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
v_initial_np = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))

u_initial_tf = tf.constant(u_initial_np, dtype=tf.float32)
v_initial_tf = tf.constant(v_initial_np, dtype=tf.float32)

# Geração de dados de referência (ground truth)
u_medido, v_medido = forward_simulation(nx, ny, nt_medido, dx, dy, dt, tf.constant(nu_real, dtype=tf.float32), u_initial_tf, v_initial_tf)

# --- Configuração do Problema Inverso (Descoberta de Parâmetros) ---

# A variável a ser descoberta (o coeficiente de difusão)
# Começamos com uma estimativa inicial (chute)
nu_guess = tf.Variable(0.1, dtype=tf.float32, name="nu_guess")
print(f"Valor inicial de nu (chute): {nu_guess.numpy()}")

# Otimizador para minimizar a função de perda
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Loop de treinamento
for epoch in range(300): # Número de épocas de otimização
    with tf.GradientTape() as tape:
        # Executa a simulação forward com o valor atual de nu_guess
        u_predicted, v_predicted = forward_simulation(
            nx, ny, nt_treino, dx, dy, dt, nu_guess, u_initial_tf, v_initial_tf
        )

        # Função de perda (Mean Squared Error)
        loss = tf.reduce_mean(tf.square(u_predicted - u_medido)) + tf.reduce_mean(tf.square(v_predicted - v_medido))
        
    # Calcula os gradientes da perda em relação a nu_guess
    gradients = tape.gradient(loss, nu_guess)
    
    # Aplica os gradientes para atualizar nu_guess
    optimizer.apply_gradients([(gradients, nu_guess)])
    
    if epoch % 50 == 0:
        print(f"Época {epoch}: Perda = {loss.numpy():.6f}, nu = {nu_guess.numpy():.6f}")

print("-" * 50)
print(f"Valor real de nu: {nu_real}")
print(f"Valor descoberto de nu: {nu_guess.numpy():.6f}")

# --- Visualização dos Resultados ---
u_final_predito, v_final_predito = forward_simulation(
    nx, ny, nt_treino, dx, dy, dt, nu_guess, u_initial_tf, v_initial_tf
)

fig = plt.figure(figsize=(11, 7), dpi=100)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.set_title("Solução Medida (nu real)")
surf1 = ax1.plot_surface(X, Y, u_medido.numpy(), cmap=cm.viridis)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('u (velocidade em x)')

ax2.set_title(f"Solução Descoberta (nu = {nu_guess.numpy():.4f})")
surf2 = ax2.plot_surface(X, Y, u_final_predito.numpy(), cmap=cm.viridis)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('u (velocidade em x)')

plt.tight_layout()
plt.savefig('burgers2d_results.pdf')
 
