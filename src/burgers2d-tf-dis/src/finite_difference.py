import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # Importa o colormap para a visualização

# Este programa resolve as equações de Burgers 2D (método explícito)
# Equações:
# u_t + u*u_x + v*u_y - nu*(u_xx + u_yy) = 0
# v_t + u*v_x + v*v_y - nu*(v_xx + v_yy) = 0

# --- Parâmetros do Problema ---
nx = 41 # Número de pontos na direção x
ny = 41 # Número de pontos na direção y
nt = 100 # Número de passos de tempo
nu = 0.01 # Coeficiente de difusão (viscosidade cinemática)
dx = 2 / (nx - 1) # Tamanho do passo espacial em x
dy = 2 / (ny - 1) # Tamanho do passo espacial em y
dt = 0.001 # Tamanho do passo de tempo

# Verificação do critério de estabilidade CFL (Courant-Friedrichs-Lewy)
# Para um método explícito como este, o passo de tempo deve ser pequeno o suficiente.
# Neste caso, a estabilidade é controlada por:
# dt <= (dx*dy) / (2*nu*(dx+dy) + max_u*dy + max_v*dx)
# Como a velocidade máxima é desconhecida a priori, ajustamos o dt manualmente.
# O valor escolhido de 0.001 é suficientemente pequeno para este problema.
# Uma verificação mais robusta poderia ser adicionada no loop de tempo.

# --- Inicialização da Grade e das Condições Iniciais ---

# Criação das grades de posição
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Condição inicial para u e v
# Vamos usar um pulso gaussiano como condição inicial em uma região da grade.
# O resto da grade tem velocidade zero.
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

# Posição do centro do pulso
center_x, center_y = 1.0, 1.0
sigma_x, sigma_y = 0.25, 0.25

# Criação do pulso gaussiano
u_pulse = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
v_pulse = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))

# Atribuição do pulso às velocidades iniciais
u = u_pulse
v = v_pulse

# Condições de contorno (Dirichlet): as bordas da grade têm velocidade zero.
# Como já inicializamos com zeros nas bordas, não precisamos fazer nada aqui.
# No loop, as bordas não serão atualizadas.

# --- Laço Principal de Resolução (Time-Stepping) ---
for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()

    # Itera sobre os pontos internos da grade para aplicar a fórmula de diferenças finitas
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Termos da equação para 'u'
            u_t = (u[j, i] - un[j, i]) / dt
            u_x = (un[j, i] - un[j, i - 1]) / dx # upwind backward em x
            u_y = (un[j, i] - un[j - 1, i]) / dy # upwind backward em y
            u_xx = (un[j, i + 1] - 2 * un[j, i] + un[j, i - 1]) / dx**2
            u_yy = (un[j + 1, i] - 2 * un[j, i] + un[j - 1, i]) / dy**2

            # Termos da equação para 'v'
            v_t = (v[j, i] - vn[j, i]) / dt
            v_x = (vn[j, i] - vn[j, i - 1]) / dx # upwind backward em x
            v_y = (vn[j, i] - vn[j - 1, i]) / dy # upwind backward em y
            v_xx = (vn[j, i + 1] - 2 * vn[j, i] + vn[j, i - 1]) / dx**2
            v_yy = (vn[j + 1, i] - 2 * vn[j, i] + vn[j - 1, i]) / dy**2

            # Atualização dos valores de 'u' e 'v' usando o esquema explícito
            u[j, i] = (un[j, i] - 
                       dt * (un[j, i] * u_x + vn[j, i] * u_y) +
                       dt * nu * (u_xx + u_yy))
            
            v[j, i] = (vn[j, i] - 
                       dt * (un[j, i] * v_x + vn[j, i] * v_y) +
                       dt * nu * (v_xx + v_yy))

# --- Visualização dos Resultados ---

# Cria a figura e os subplots para visualização
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d') # Cria um subplot 3D

# Plota a superfície da velocidade 'u'
surf = ax.plot_surface(X, Y, u, cmap=cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u (velocidade em x)')
ax.set_title('Distribuição de Velocidade u(x, y, t) no Tempo Final')

plt.savefig('burgers2d_diff_results.jpg')