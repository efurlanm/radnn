# Descrição do Arquivo `burgers2d.py`

Este script Python implementa a solução de um problema inverso para a equação de Burgers 2D, utilizando o framework TensorFlow para a descoberta de parâmetros. O objetivo principal é determinar o coeficiente de viscosidade cinemática (`nu`) da equação de Burgers 2D a partir de dados de simulação "medidos" (gerados internamente pelo próprio script).

## Equações de Burgers 2D

O problema aborda um sistema de duas equações diferenciais parciais acopladas para as componentes de velocidade `u` (direção x) e `v` (direção y), incluindo um termo de viscosidade:

1. **Componente `u` (velocidade na direção x):**
   `u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) = 0`

2. **Componente `v` (velocidade na direção y):**
   `v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) = 0`

Onde:

- `u` e `v` são as componentes da velocidade.
- `u_t`, `u_x`, `u_y`, `v_t`, `v_x`, `v_y` são as derivadas parciais de primeira ordem em relação ao tempo (`t`) e às coordenadas espaciais (`x`, `y`).
- `u_xx`, `u_yy`, `v_xx`, `v_yy` são as derivadas parciais de segunda ordem em relação a `x` e `y`.
- `nu` é o coeficiente de viscosidade cinemática, o parâmetro a ser descoberto.

## Estrutura do Script

O script pode ser dividido nas seguintes seções principais:

### 1. Importações e Parâmetros Iniciais

- **Importações:** `numpy` para operações numéricas, `tensorflow` para construção do modelo e otimização, `matplotlib.pyplot` e `matplotlib.cm` para visualização.
- **Parâmetros do Problema:**
  - `nx`, `ny`: Número de pontos na grade espacial (41 em ambas as direções).
  - `nt_medido`: Número de passos de tempo para a simulação de dados "medidos" (50).
  - `nt_treino`: Número de passos de tempo para a simulação durante o treinamento (deve ser igual a `nt_medido`, também 50).
  - `dx`, `dy`: Tamanho do passo espacial.
  - `dt`: Tamanho do passo de tempo (0.001).

### 2. Geração de Dados "Medidos" (Solução de Referência)

- Uma função `forward_simulation` é definida para simular a equação de Burgers 2D usando um método explícito de diferenças finitas.
- Esta função aceita um valor de `nu` (que pode ser um tensor TensorFlow para permitir o cálculo de gradientes) e condições iniciais para `u` e `v`.
- **Cálculo de Derivadas:** As derivadas espaciais de primeira e segunda ordem são calculadas usando diferenças finitas centrais.
- **Termos de Convecção e Difusão:** Os termos não lineares de convecção (`u*u_x + v*u_y` e `u*v_x + v*v_y`) e os termos de difusão (`nu*(u_xx + u_yy)` e `nu*(v_xx + v_yy)`) são calculados.
- **Atualização Explícita:** As equações são atualizadas explicitamente no tempo.
- **Condições Iniciais:** As condições iniciais para `u` e `v` são definidas como funções gaussianas 2D, centralizadas em (1.0, 1.0).
- **Geração de Dados de Referência:** A simulação direta é executada com um `nu_real` conhecido (0.05) para gerar os dados `u_medido` e `v_medido`, que servem como a "verdade" ou "observação" para o problema inverso.

### 3. Configuração do Problema Inverso (Descoberta de Parâmetros)

- **Variável a ser Descoberta:** `nu_guess` é inicializado como uma variável TensorFlow (`tf.Variable`) com um chute inicial (0.1). Esta é a variável que o otimizador ajustará.
- **Otimizador:** Um otimizador `tf.keras.optimizers.Adam` é configurado com uma taxa de aprendizado de 0.01.
- **Loop de Treinamento:**
  - O script executa um loop de treinamento por 500 épocas.
  - Dentro de cada época:
    - Uma `tf.GradientTape` é usada para registrar as operações e calcular gradientes.
    - A função `forward_simulation` é chamada com o `nu_guess` atual para obter `u_predicted` e `v_predicted`.
    - **Função de Perda (Loss Function):** A perda é calculada como o Erro Quadrático Médio (MSE) entre as soluções preditas (`u_predicted`, `v_predicted`) e as soluções medidas (`u_medido`, `v_medido`).
    - Os gradientes da perda em relação a `nu_guess` são calculados.
    - O otimizador aplica esses gradientes para atualizar `nu_guess`, minimizando a perda.
    - O progresso (perda e valor atual de `nu_guess`) é impresso a cada 50 épocas.

### 4. Visualização dos Resultados

- Após o treinamento, a simulação direta é executada novamente com o `nu_guess` final para obter `u_final_predito` e `v_final_predito`.
- Duas subtramas são geradas usando `matplotlib`:
  - Uma mostra a solução `u_medido` (com o `nu_real`).
  - A outra mostra a solução `u_final_predito` (com o `nu_guess` descoberto).
- Mapas de contorno (`contourf`) são usados para visualizar as distribuições de velocidade, e barras de cores são adicionadas para referência.
- Os gráficos são exibidos para comparação visual entre a solução real e a solução predita com o parâmetro descoberto.

## Propósito

O script demonstra uma abordagem de "Physics-Informed Neural Networks" (PINN) ou, mais precisamente, uma otimização baseada em gradiente para a descoberta de parâmetros em PDEs. Em vez de uma rede neural completa, ele usa a simulação direta da PDE dentro de um loop de otimização do TensorFlow para ajustar um parâmetro desconhecido (`nu`) até que a saída da simulação se ajuste aos dados observados.
