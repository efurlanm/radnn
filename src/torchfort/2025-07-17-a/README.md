# Snapshot do Progresso - Conversão Burgers 1D (PyTorch para TorchFort)

**Data:** 2025-07-17

## Visão Geral

Este diretório contém um snapshot do progresso atual na conversão do problema de Burgers 1D de PyTorch para TorchFort. Ele demonstra a geração de dados, treinamento e inferência em Python, e o treinamento em Fortran, com verificação de consistência.

## O que funciona

* **Geração de Dados (Python):**
  * O script `generate_burgers_model.py` gera os pontos de treinamento (collocation points, initial conditions, boundary conditions) e os salva em arquivos de texto (`X_f.txt`, `x0_t0.txt`, `u0.txt`, `xb_left_tb.txt`, `xb_right_tb.txt`).
  * Ele também gera os modelos TorchScript iniciais (`burgers_model.pt`, `burgers_loss.pt`).
* **Treinamento em Python:**
  * O script `burgers1d_original.py` treina o modelo PINN usando PyTorch.
  * Ele salva o `state_dict` do modelo treinado em `pinn_model_state.pt`.
  * A saída de inferência direta do modelo Python treinado é salva em `original_python_u_pred_direct.txt`.
  * Uma nova instância do modelo `PINN` é criada, o `state_dict` é carregado nela, e esta instância *treinada* é então scriptada e salva como `burgers_inference_model.pt`. Este é o modelo TorchScript que será usado para inferência em Fortran.
* **Inferência em Python:**
  * O script `burgers_inference.py` carrega o `pinn_model_state.pt` (o `state_dict` do modelo treinado).
  * Ele executa a inferência usando este modelo carregado e salva a saída em `python_inference_u_pred.txt`.
* **Comparação de Inferência Python-para-Python:**
  * O script `compare_inference_results.py` compara `original_python_u_pred_direct.txt` (saída direta do treinamento Python) com `python_inference_u_pred.txt` (saída da inferência Python usando o `state_dict` carregado).
  * **Status:** Esta comparação mostra consistência total (diferença máxima e média de 0.00e+00), confirmando que o processo de salvamento/carregamento do modelo em Python e a inferência são robustos.
* **Treinamento em Fortran:**
  * O script `burgers_train.f90` compila e executa com sucesso, lendo os dados de texto gerados pelo Python e treinando o modelo.
  * Ele salva o modelo treinado em `burgers_model_trained.pt`.
  * O número de épocas de treinamento é padronizado para 5000 para testes rápidos.

## Estrutura do Diretório

```
2025-07-17-a/
├── generate_burgers_model.py
├── burgers1d_original.py
├── burgers_inference.py
├── compare_inference_results.py
├── compare_data.py
├── test_pinn_inference.py
├── burgers_train.f90
├── burgers_inference.f90
├── burgers_train_config.yaml
└── CMakeLists.txt
```

## Como Reproduzir

**Pré-requisitos:**

* Ambiente Singularity com o container `~/containers/torchfort.sif` disponível.
* O ambiente TorchFort deve estar configurado dentro do container.

**Passos para Reprodução (executar a partir do diretório `examples/fortran/burgers/2025-07-17-a/`):**

1. **Gerar Dados e Modelos Iniciais (Python):**
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "cd /torchfort/examples/fortran/burgers/2025-07-17-a && python3 generate_burgers_model.py"
   ```
   
   * *Saídas:* `X_f.txt`, `x0_t0.txt`, `u0.txt`, `xb_left_tb.txt`, `xb_right_tb.txt`, `burgers_model.pt`, `burgers_loss.pt`.

2. **Treinar Modelo e Gerar Referência de Inferência (Python):**
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "cd /torchfort/examples/fortran/burgers/2025-07-17-a && python3 burgers1d_original.py"
   ```
   
   * *Saídas:* `pinn_model_state.pt`, `original_python_u_pred_direct.txt`, `burgers_inference_model.pt`.

3. **Executar Inferência (Python, usando state_dict):**
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "cd /torchfort/examples/fortran/burgers/2025-07-17-a && python3 burgers_inference.py"
   ```
   
   * *Saída:* `python_inference_u_pred.txt`.

4. **Comparar Inferência Python-para-Python:**
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "cd /torchfort/examples/fortran/burgers/2025-07-17-a && python3 compare_inference_results.py"
   ```
   
   * *Saída:* Relatório de comparação e `python_inference_comparison.png`.

5. **Compilar Executáveis Fortran:**
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && cd /torchfort/build && make burgers_train burgers_inference"
   ```

6. **Copiar Executáveis Fortran:**
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "cp /torchfort/build/examples/fortran/burgers/burgers_train /torchfort/examples/fortran/burgers/2025-07-17-a/"
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "cp /torchfort/build/examples/fortran/burgers/burgers_inference /torchfort/examples/fortran/burgers/2025-07-17-a/"
   ```

7. **Executar Treinamento (Fortran):**
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && cd /torchfort/examples/fortran/burgers/2025-07-17-a && ./burgers_train"
   ```
   
   * *Saída:* `burgers_model_trained.pt`.

**Próximos Passos (a serem implementados):**

* Executar Inferência (Fortran).
* Comparar Inferência Fortran-para-Python.
