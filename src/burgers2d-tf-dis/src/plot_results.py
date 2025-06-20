import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os

# --- Argumentos de Linha de Comando ---
# Define o arquivo de entrada padrão
default_input_file = 'pinn_results_03_scipy_test.npz'

# Pega o nome do arquivo do argumento de linha de comando, se fornecido
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = default_input_file

# Verifica se o arquivo de entrada existe
if not os.path.exists(input_file):
    print(f"Erro: Arquivo de entrada '{input_file}' não encontrado.")
    sys.exit(1)

# Gera o nome do arquivo de saída a partir do nome do arquivo de entrada
output_filename = os.path.splitext(os.path.basename(input_file))[0] + '.jpg'
output_path = os.path.splitext(os.path.basename(input_file))[0] + '.jpg'

# --- Carregar os Resultados ---
print(f"Carregando resultados de: {input_file}")
data = np.load(input_file)
X_np = data['X']
Y_np = data['Y']
u_medido = data['u_medido']
u_pinn_pred = data['u_pinn_pred']
nu_pinn = data['nu_pinn']

# --- Visualização ---
fig = plt.figure(figsize=(11, 7), dpi=100)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.set_title("Solução Medida (nu real)")
surf1 = ax1.plot_surface(X_np, Y_np, u_medido, cmap=cm.viridis)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('u (velocidade em x)')

ax2.set_title(f"Solução PINN (nu = {nu_pinn:.4f})")
surf2 = ax2.plot_surface(X_np, Y_np, u_pinn_pred, cmap=cm.viridis)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('u (velocidade em x)')

plt.tight_layout()
plt.savefig(output_path)

print(f"Gráfico salvo em: {output_path}")

print("\n--- Statistics ---")
print(f"u_medido:    min={np.min(u_medido):.4f}, max={np.max(u_medido):.4f}, mean={np.mean(u_medido):.4f}")
print(f"u_pinn_pred: min={np.min(u_pinn_pred):.4f}, max={np.max(u_pinn_pred):.4f}, mean={np.mean(u_pinn_pred):.4f}")
