# Análise Detalhada do Script `burgers1d.py`

Este documento descreve o funcionamento do script Python `burgers1d.py`, que utiliza uma Rede Neural Informada pela Física (PINN) para resolver a equação de Burgers 1D.

## 1. Visão Geral

O script implementa uma solução baseada em `PyTorch` para a equação diferencial parcial (EDP) de Burgers:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}
$$

Onde:
- `u(x, t)` é a função de velocidade.
- `ν` (nu) é o coeficiente de viscosidade.

A abordagem PINN consiste em treinar uma rede neural para que ela não apenas se ajuste aos dados conhecidos (condições iniciais e de contorno), mas também obedeça à EDP governante.

## 2. Estrutura do Código

### 2.1. Importações e Configuração Inicial

O script começa importando as bibliotecas necessárias e definindo uma semente para reprodutibilidade.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Fixed random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set the default tensor type to float32
torch.set_default_dtype(torch.float32)
```

- **`torch`**: Biblioteca principal para deep learning.
- **`numpy`**: Usada para manipulação de arrays numéricos.
- **`matplotlib.pyplot`**: Para visualização dos resultados.
- A semente (`SEED`) garante que os resultados aleatórios (como a inicialização de pesos da rede e a amostragem de pontos) sejam os mesmos a cada execução.
- `torch.set_default_dtype(torch.float32)` define o tipo de tensor padrão, o que é importante para consistência numérica.

### 2.2. Parâmetros Físicos e de Amostragem

Define-se o domínio do problema e os parâmetros de amostragem.

```python
# Define physical parameters and boundaries
x_min, x_max = -1.0, 1.0  # Spatial range
t_min, t_max = 0.0, 1.0   # Temporal range
nu = 0.01 / np.pi         # Diffusion coefficient

# Define the number of sampled points
N_f = 10000  # Collocation points for the PDE
N_0 = 400    # Initial condition points
N_b = 200    # Boundary condition points
```

- **Domínio**: O problema é resolvido no espaço `x` de `[-1, 1]` e no tempo `t` de `[0, 1]`.
- **`nu`**: Coeficiente de viscosidade (ou difusão).
- **Pontos de Amostragem**:
    - `N_f`: Número de pontos de colocação, amostrados aleatoriamente dentro do domínio `(x, t)`, onde o resíduo da EDP será minimizado.
    - `N_0`: Número de pontos na condição inicial (`t=0`).
    - `N_b`: Número de pontos nas condições de contorno (`x=-1` e `x=1`).

### 2.3. Geração de Dados de Treinamento

Nesta seção, os pontos de treinamento são gerados com base nos parâmetros definidos.

- **Pontos de Colocação (`X_f`)**:
  ```python
  X_f = np.random.rand(N_f, 2)
  X_f[:, 0] = X_f[:, 0] * (x_max - x_min) + x_min
  X_f[:, 1] = X_f[:, 1] * (t_max - t_min) + t_min
  ```
  `N_f` pontos `(x, t)` são amostrados aleatoriamente no domínio para garantir que a rede neural aprenda a física da EDP em toda a sua extensão.

- **Condição Inicial**: `u(x, 0) = -sin(πx)`
  ```python
  x0 = np.linspace(x_min, x_max, N_0)[:, None]
  t0 = np.zeros_like(x0)
  u0 = -np.sin(np.pi * x0)
  ```
  `N_0` pontos são criados ao longo do eixo `x` no tempo `t=0`, com o valor de `u` correspondente.

- **Condições de Contorno**: `u(-1, t) = 0` e `u(1, t) = 0`
  ```python
  tb = np.linspace(t_min, t_max, N_b)[:, None]
  xb_left = np.ones_like(tb) * x_min
  xb_right = np.ones_like(tb) * x_max
  ub_left = np.zeros_like(tb)
  ub_right = np.zeros_like(tb)
  ```
  `N_b` pontos são criados ao longo do tempo `t` nas fronteiras espaciais `x=-1` e `x=1`, onde o valor de `u` é zero.

### 2.4. Conversão para Tensores e Definição do Dispositivo

Os dados gerados com `numpy` são convertidos para tensores `PyTorch` e movidos para a GPU, se disponível.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(device)
# ... (outras conversões)
```
- `requires_grad=True` para `X_f` é crucial, pois permite o cálculo de derivadas parciais de `u` em relação a `x` e `t` usando o `autograd` do PyTorch.

### 2.5. Definição da Rede Neural (PINN)

A arquitetura da rede neural é definida como uma classe `PINN`.

```python
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

layers = [2, 10, 10, 10, 10, 10, 10, 10, 10, 1]
model = PINN(layers).to(device)
```
- A rede é um Perceptron de Múltiplas Camadas (MLP) simples.
- **Entrada**: Um tensor de tamanho `[N, 2]` representando os pontos `(x, t)`.
- **Saída**: Um tensor de tamanho `[N, 1]` representando a predição `u(x, t)`.
- **Arquitetura**: 1 camada de entrada (2 neurônios), 8 camadas ocultas (10 neurônios cada) e 1 camada de saída (1 neurônio).
- **Função de Ativação**: Tangente Hiperbólica (`Tanh`).

### 2.6. Função de Resíduo da EDP

Esta é a parte "informada pela física". A função `pde_residual` calcula o quanto a predição da rede viola a equação de Burgers.

```python
def pde_residual(model, X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    u = model(torch.cat([x, t], dim=1))

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    f = u_t + u * u_x - nu * u_xx
    return f
```
- `torch.autograd.grad` é usado para calcular as derivadas parciais (`u_t`, `u_x`, `u_xx`) de forma automática.
- `f` representa o resíduo da EDP. O objetivo do treinamento é forçar `f` a ser o mais próximo possível de zero.

### 2.7. Função de Perda (Loss)

A função de perda total combina o erro da EDP, o erro da condição inicial e o erro das condições de contorno.

```python
def loss_func(model):
    loss_f = torch.mean(pde_residual(model, X_f)**2)
    loss_0 = torch.mean((model(torch.cat([x0, t0], dim=1)) - u0)**2)
    loss_b = torch.mean(model(torch.cat([xb_left, tb], dim=1))**2) + torch.mean(model(torch.cat([xb_right, tb], dim=1))**2)
    return loss_f + loss_0 + loss_b
```
- **`loss_f`**: Erro quadrático médio do resíduo da EDP nos pontos de colocação. Força a rede a obedecer à física.
- **`loss_0`**: Erro quadrático médio na condição inicial. Força a rede a corresponder à solução em `t=0`.
- **`loss_b`**: Erro quadrático médio nas condições de contorno. Força a rede a corresponder à solução em `x=-1` e `x=1`.

A perda total é a soma desses três componentes, que são minimizados simultaneamente durante o treinamento.

### 2.8. Treinamento

Um loop de otimização padrão é usado para treinar o modelo.

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_func(model)
    loss.backward()
    optimizer.step()
    # ... (impressão do progresso)
```
- **Otimizador**: `Adam` é um otimizador popular e eficaz para muitas tarefas de deep learning.
- **Loop**: A cada época, os gradientes da perda em relação aos parâmetros do modelo são calculados (`loss.backward()`) e os parâmetros são atualizados (`optimizer.step()`).

### 2.9. Avaliação, Visualização e Salvamento

Após o treinamento, o modelo é avaliado em uma grade regular de pontos `(x, t)` para visualização.

```python
# Generate a grid for visualization
N_x, N_t = 256, 100
x = np.linspace(x_min, x_max, N_x)
t = np.linspace(t_min, t_max, N_t)
X, T = np.meshgrid(x, t)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
XT_tensor = torch.tensor(XT, dtype=torch.float32).to(device)

# Compute the predicted solution
with torch.no_grad():
    u_pred = model(XT_tensor).cpu().numpy().reshape(N_t, N_x)

# Save results to a binary file
output_filename = "burgers1d_python_original_results.bin"
with open(output_filename, 'wb') as f:
    # ... (código para salvar)

# Plot the predicted solution
plt.figure(figsize=(8, 5))
plt.contourf(X, T, u_pred, levels=100, cmap='viridis')
# ... (código de plotagem)
plt.show()
```
- O modelo treinado (`model.eval()`) é usado para prever a solução `u_pred` em toda a grade.
- Os resultados (grade `x`, `t` e a solução `u_pred`) são salvos em um arquivo binário para comparações futuras.
- A solução é visualizada como um mapa de contorno usando `matplotlib`.
