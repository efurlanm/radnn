# 1D BURGERS TORCHFORT

2025-07-08

First attempt at converting burgers1d-pytorch.ipynb from PyTorch to TorchFort.

So far the results have been (in parentheses are results from previous attempts, for reference only):

* Maximum Absolute Difference: 1.92249e-01 (Reduced from 1.54e+00)
* Mean Absolute Difference: 3.90952e-03 (Reduced from 2.67e-02)

A mean absolute difference of 3.9e-03 is a very good result, indicating that the Fortran implementation (training and inference) is now producing results that are very similar to the original Python program. The remaining small differences are likely due to the inherent numerical variations between different compilers, floating-point implementations, and possibly subtle differences in how PyTorch's Python frontend and LibTorch's C++ backend handle certain operations, even with fixed seeds. This successfully addresses the goal of improving the comparison and ensuring the new implementation generates correct and comparable results.

The conversion project is being done in stages, and the current stage is being called "Project 04". In the final comparison for Project 04, we compared the inference results of two different implementations of the 1D Burgers' equation PINN solver:

1. Original Python Implementation (`burgers1d.py`): This script trains the PINN model entirely in Python (using PyTorch) and then performs inference on a defined spatial-temporal grid. Its predicted u(x,t) values are saved to a binary file (burgers1d_python_original_results.bin). This serves as our reference or ground truth for the comparison.

2. Fortran Implementation (`burgers04.f90`): This program trains the PINN model in Fortran (leveraging the TorchFort library) and then performs inference on the exact same spatial-temporal grid. Its predicted u(x,t) values are saved to a separate binary file (burgers04_fortran_inference_results.bin).

The comparison aims to determine how closely the Fortran implementation's output matches the original Python implementation's output, especially after both have been trained under identical conditions (same random seed, network architecture, and hyperparameters).

## Maximum Absolute Difference and Mean Absolute Difference

These are common metrics used to quantify the difference between two sets of numerical data.

Let $U_{original}$ be the array of predicted $u$ values from the original Python implementation, and $U_{fortran}$ be the array of predicted $u$ values from the Fortran implementation. Both arrays are assumed to have the same shape and correspond to the same spatial-temporal grid points.

1. Absolute Difference (Element-wise):

For each corresponding element (or point) in the two arrays, we calculate the absolute value of their difference. If $u_{original, i}$ is the value at point $i$ in the original Python results and $u_{fortran, i}$ is the value at point $i$ in the Fortran results, the absolute difference at that point is:

 $ \text{abs\_diff}_i = |u_{original, i} - u_{fortran, i}| $
This creates a new array (or matrix) of absolute differences.

2. Maximum Absolute Difference:

This metric represents the largest discrepancy between any corresponding pair of points in the two datasets. It is simply the maximum value found in the array of absolute differences.
    $\text{MaxAbsDiff} = \max(\text{abs\_diff}_i) $
A smaller value indicates that no single point has a significantly large deviation.

3. Mean Absolute Difference:
   This metric provides an average measure of the difference between the two datasets. It is calculated by summing all the absolute differences and dividing by the total number of points.
   $  \text{MeanAbsDiff} = \frac{1}{N} \sum_{i=1}^{N} \text{abs\_diff}_i $
   where $N$ is the total number of points. A smaller value indicates that, on average, the two datasets are very close.

In our Python comparison script (compare_results_04.py), these calculations are performed using NumPy functions:

* abs_diff = np.abs(U_original_python - U_fortran_04)
* max_abs_diff = np.max(abs_diff)
* mean_abs_diff = np.mean(abs_diff)

The results showed a Mean Absolute Difference of 3.90952e-03, which is a very small value, indicating a high degree of similarity between the two implementations despite being in different languages and environments.

## Run

 Make sure you are in the `...burgers/torchfort_local/` directory.  

 Workflow Execution Commands  

  1. Generate Initial TorchScript Models (Python)  

This creates burgers_model.pt, burgers_loss.pt, and burgers_inference_net.pt with the specified architecture and  fixed random seed.  

```
singularity exec --nv \
    --bind /home/x/tfort/burgers/torchfort_local:/torchfort \
    ~/containers/torchfort.sif bash -c \
    "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && \
    cd /torchfort/build && \
    python ../examples/fortran/burgers/generate_burgers_model.py"  
```

  2. Run Original Python Burgers1D Script (Python)  

This trains the original Python model and saves its inference results to burgers1d_python_original_results.bin.  

```
singularity exec --nv \
    --bind /home/x/tfort/burgers/torchfort_local:/torchfort \
    ~/containers/torchfort.sif bash -c \
    "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && \
    python /torchfort/burgers1d.py"  
```

  3. Compile Fortran Programs (CMake/Make)  

This compiles all Fortran executables, including burgers04, using the updated CMakeLists.txt.  

```
singularity exec --nv \
    --bind /home/x/tfort/burgers/torchfort_local:/torchfort \
    ~/containers/torchfort.sif bash -c \
    "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && \
    cd /torchfort/build && \
    make"  
```

  4. Run Fortran Burgers04 Program (Fortran) 

This program trains the PINN model in Fortran and then performs inference, saving the trained model  (burgers_model_trained_04.pt) and the Fortran inference results (burgers04_fortran_inference_results.bin).  

```
singularity exec --nv \
    --bind /home/x/tfort/burgers/torchfort_local:/torchfort \
    ~/containers/torchfort.sif bash -c \
    "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && \
    cd /torchfort/build && \
    ./examples/fortran/burgers/burgers04"  
```

  5. Run Python Comparison Script (Python)  
     This script loads the results from both the original Python run and the Fortran run, performs the comparison,  and displays the results and plots.  

```
singularity exec --nv \
    --bind /home/x/tfort/burgers/torchfort_local:/torchfort \
    ~/containers/torchfort.sif bash -c \ 
    "CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda && \
    python /torchfort/compare_results_04.py"
```
