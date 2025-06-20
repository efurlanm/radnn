# Snapshot: 2025-07-17-c - Validação do Treinamento Fortran

Este diretório contém um snapshot do estado atual do projeto de conversão do modelo Burgers 1D de PyTorch para TorchFort, focado na validação da etapa de treinamento em Fortran.

## Conteúdo do Snapshot

- `burgers1d_original.py`: O script original em PyTorch para treinamento e inferência do modelo Burgers 1D. Serve como referência para a arquitetura do modelo e os resultados esperados.
- `config_burgers.yaml`: O arquivo de configuração YAML para o modelo MLP usado pelo TorchFort, adaptado para a arquitetura do modelo PINN.
- `burgers_train_02.f90`: O programa Fortran que realiza o treinamento do modelo usando TorchFort e, em seguida, executa a inferência com o modelo treinado, salvando os resultados.
- `CMakeLists.txt`: O arquivo de configuração do CMake, modificado para incluir a compilação de `burgers_train_02.f90`.
- `compare_u_pred_results_02.py`: Um script Python para comparar os resultados de `u_pred` gerados pelo `burgers1d_original.py` (Python) e pelo `burgers_train_02.f90` (Fortran).
- `burgers_model_trained.pt`: O modelo TorchScript salvo após o treinamento bem-sucedido em Fortran via `burgers_train_02.f90`.
- `fortran_trained_u_pred_02.bin`: O arquivo binário contendo os resultados da inferência (`N_x`, `N_t`, `x`, `t`, `u_pred`) gerados pelo `burgers_train_02.f90`.
- `burgers1d_python_original_results.bin`: O arquivo binário contendo os resultados da inferência (`N_x`, `N_t`, `x`, `t`, `u_pred`) gerados pelo `burgers1d_original.py`.

## O que funciona

1. **Treinamento em PyTorch (`burgers1d_original.py`)**: O script original em Python treina o modelo PINN com sucesso e salva seus resultados de inferência em `burgers1d_python_original_results.bin`.
2. **Compilação do Fortran**: O programa Fortran `burgers_train_02.f90` compila com sucesso após as modificações no `CMakeLists.txt` e no `config_burgers.yaml`.
3. **Treinamento em Fortran (`burgers_train_02.f90`)**: O treinamento do modelo em Fortran é executado com sucesso, e a perda diminui ao longo das épocas. O modelo treinado é salvo como `burgers_model_trained.pt`.
4. **Inferência em Fortran (`burgers_train_02.f90`)**: A inferência é realizada com sucesso dentro do programa Fortran após o treinamento, e os resultados são salvos em `fortran_trained_u_pred_02.bin`.
5. **Comparação de Dimensões e Eixos**: O script `compare_u_pred_results_02.py` confirma que as dimensões da grade (`N_x`, `N_t`) e os arrays `x` e `t` são numericamente similares entre as saídas Python e Fortran.

## O que não funciona e por quê

1. **Similaridade Numérica de `u_pred`**: A principal divergência é que os valores de `u_pred` (a solução predita) do modelo treinado e inferido em Fortran **não são numericamente similares** aos do modelo original em Python. A diferença absoluta máxima é de aproximadamente `9.7e-01`, e a diferença absoluta média é de `5.2e-01`, o que é muito alto para ser aceitável.
   
   **Razão**: A causa raiz dessa divergência reside na implementação da função de perda no Fortran. O modelo PINN original em Python utiliza uma função de perda complexa que envolve a diferenciação automática da saída do modelo em relação às suas entradas (`x` e `t`) para calcular o resíduo da Equação Diferencial Parcial (PDE) e as condições de contorno. No `burgers_train_02.f90`, a função de perda atual (`MSELoss` configurada no `config_burgers.yaml`) simplesmente minimiza o erro quadrático médio entre a saída do modelo e rótulos pré-calculados (zeros para a PDE e valores iniciais/de contorno para as condições). Isso **não replica** a lógica de diferenciação automática necessária para a perda da PDE, resultando em um treinamento incorreto do modelo em Fortran para o problema de PINN.

2. **Carregamento do Modelo Treinado em Fortran no Python**: Tentativas anteriores de carregar o modelo `burgers_model_trained.pt` (salvo pelo TorchFort) diretamente em Python usando `torch.jit.load` para inferência falharam com um `AttributeError: 'RecursiveScriptModule' object has no attribute 'forward'`. Isso indica que o formato TorchScript salvo pelo TorchFort pode não ser diretamente compatível ou facilmente utilizável como um `nn.Module` padrão no ambiente Python, o que levou à decisão de realizar a inferência diretamente no Fortran para a comparação atual.

## Como Executar e Reproduzir

Para reproduzir o estado atual e os resultados:

1. **Navegar para o diretório do snapshot**: 
   `cd /home/x/tfort/burgers/torchfort_local/examples/fortran/burgers/2025-07-17-c`

2. **Configurar o ambiente Singularity**: Certifique-se de que o container Singularity (`~/containers/torchfort.sif`) esteja acessível.

3. **Re-executar CMake (se necessário)**: Se você estiver em um ambiente limpo ou tiver problemas de compilação, execute o CMake para gerar os Makefiles:
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && cd /torchfort/build && cmake .."
   ```
   
   *Nota: Este comando deve ser executado a partir do diretório raiz do projeto (`/home/x/tfort/burgers/torchfort_local/`), mas o `cd /torchfort/build` garante que ele seja executado no contexto correto dentro do container.* 
   *Após executar o `cmake ..`, você precisará copiar o executável `burgers_train_02` do diretório `build` para o diretório atual do snapshot, pois o `cmake` gera os executáveis no diretório `build`.* 
   `cp /home/x/tfort/burgers/torchfort_local/build/examples/fortran/burgers/burgers_train_02 .`

4. **Compilar o programa Fortran**: Compile `burgers_train_02.f90`:
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && cd /torchfort/build && make burgers_train_02"
   ```

5. **Copiar o executável Fortran**: Copie o executável compilado para o diretório atual:
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "cp /torchfort/build/examples/fortran/burgers/burgers_train_02 /torchfort/examples/fortran/burgers/2025-07-17-c/"
   ```

6. **Executar o treinamento e inferência Fortran**: Execute o programa Fortran. Isso irá treinar o modelo e gerar o arquivo `fortran_trained_u_pred_02.bin`:
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "CUDA_PATH=/opt/nvidia/hpc_s_sdk/Linux_x86_64/25.3/cuda && cd /torchfort/examples/fortran/burgers/2025-07-17-c && ./burgers_train_02"
   ```

7. **Gerar resultados de referência em Python (se ainda não existirem)**: Execute o script Python original para gerar `burgers1d_python_original_results.bin`:
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "CUDA_PATH=/opt/nvidia/hpc_s_sdk/Linux_x86_64/25.3/cuda && cd /torchfort/examples/fortran/burgers/2025-07-17-c && python3 burgers1d_original.py"
   ```

8. **Executar o script de comparação Python**: Compare os resultados:
   
   ```bash
   singularity exec --nv --bind .:/torchfort ~/containers/torchfort.sif bash -c "CUDA_PATH=/opt/nvidia/hpc_s_sdk/Linux_x86_64/25.3/cuda && cd /torchfort/examples/fortran/burgers/2025-07-17-c && python3 compare_u_pred_results_02.py"
   ```

## Próximos Passos

O principal desafio é implementar a função de perda do PINN corretamente no ambiente TorchFort para que o treinamento em Fortran produza resultados numericamente similares aos do PyTorch.

1. **Implementar a Função de Perda PINN em Python (TorchScript)**:
   
   * Criar um novo script Python (e.g., `pinn_loss_module.py`) que define a lógica da função de perda do PINN (`pde_residual`, condições iniciais e de contorno) como um `nn.Module` ou uma função Python que pode ser exportada para TorchScript.
   * Utilizar `torch.autograd.grad` dentro desta função de perda para calcular as derivadas necessárias para a PDE.
   * Exportar esta função de perda para um arquivo TorchScript (e.g., `pinn_loss.pt`).

2. **Modificar `burgers_train_02.f90` para usar a Perda TorchScript**: 
   
   * Atualizar `config_burgers.yaml` para especificar o tipo de perda como `torchscript` e apontar para `pinn_loss.pt`.
   * Modificar `burgers_train_02.f90` para carregar esta função de perda TorchScript e passá-la para `torchfort_train_multiarg` junto com os dados de entrada e os argumentos adicionais necessários para a perda (e.g., `X_f`, `x0`, `t0`, `u0`, `xb_left`, `tb`, `xb_right`, `ub_left`, `ub_right`).

3. **Re-treinar e Re-inferir em Fortran**: Compilar e executar o `burgers_train_02.f90` modificado.

4. **Re-comparar Resultados**: Executar `compare_u_pred_results_02.py` novamente para verificar se a divergência numérica foi resolvida e se os resultados são agora similares.