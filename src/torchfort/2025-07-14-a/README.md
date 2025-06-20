# Project State: Burgers' Equation PINN Conversion to TorchFort (2025-07-14)

This directory captures the state of the project at a specific point in time, focusing on the conversion of a PyTorch-based Physics-Informed Neural Network (PINN) for the Burgers' equation to a TorchFort-compatible framework. The objective is to achieve functional equivalence across Python (PyTorch) and Fortran (TorchFort) implementations for both model training and inference.

## Current Status Overview

As of this snapshot, the project has successfully demonstrated:

1.  **Robust PyTorch-side TorchScript Model Generation for Inference:**
    *   The `burgers1d.py` script has been meticulously refined to train the PINN model and subsequently export a TorchScript model (`burgers_inference_trained.pt`) specifically optimized for inference. This was achieved by instantiating a new `PINN` model after the training phase, loading the trained `state_dict` into this dedicated instance, and then applying `torch.jit.script` to this clean, inference-focused model. This methodology ensures that the exported TorchScript graph's `forward` method strictly adheres to a single-input signature, accepting only the `(x, t)` coordinate tensor.
    *   **Verification of Python-side Inference:** A dedicated Python script, `examples/fortran/burgers/inference_test.py`, was developed to load `burgers_inference_trained.pt` and execute inference. A subsequent comparison script, `examples/fortran/burgers/compare_python_results.py`, was utilized to perform a rigorous numerical comparison between the output of `inference_test.py` and the original inference results generated directly by `burgers1d.py`. The comparison yielded a maximum absolute difference of `0.00e+00` and a mean absolute difference of `0.00e+00`. This result conclusively validates the correctness of the Python-side TorchScript export and the fidelity of the inference process when re-importing the scripted model within the Python environment.

2.  **Functional Fortran Training Module:**
    *   The Fortran training program, `examples/fortran/burgers/burgers_train.f90`, has been successfully compiled and executed within the Singularity container environment. This module effectively leverages the TorchFort API to train the PINN model using the defined physics-informed loss function.
    *   Initial challenges related to data layout discrepancies between Fortran's column-major array allocation and PyTorch's row-major tensor expectations were addressed by transposing the input data arrays in Fortran to conform to a `(features, N)` dimension.
    *   The training process consistently demonstrates convergence, with final loss values (e.g., approximately `6.84e-03`) that are quantitatively comparable to those observed during the original PyTorch training. This confirms the proper integration and operational integrity of the TorchFort training API.

## Outstanding Issue: Fortran Inference Failure

Despite the successful Python-side generation and verification of a single-input TorchScript model, the Fortran inference program (`examples/fortran/burgers/burgers_inference.f90`) continues to encounter a persistent runtime error during the `torchfort_inference_multiarg` call. The error message consistently reports: `forward() is missing value for argument 'x0_cat'. Declaration: forward(__torch__.BurgersPINN self, Tensor x_f, Tensor x0_cat, Tensor xb_left_cat, Tensor xb_right_cat) -> Tensor`.

This error is critical because it indicates that the TorchFort runtime, when attempting to load and execute `burgers_inference_trained.pt`, is still expecting a multi-argument `forward` method signature. This signature corresponds to the `BurgersPINN` class's comprehensive training-oriented `forward` method, rather than the simplified single-input `forward` method of the `PINN` class (which is the intended inference model). This discrepancy suggests a potential issue in how TorchFort internally interprets or loads the TorchScript graph, or a subtle interaction within the TorchFort API's mechanism for handling dynamically traced models.

## Next Steps for Resolution

The immediate priority is to conduct a direct, low-level inspection of the `burgers_inference_trained.pt` file using the `old/inspect_model.py` script (which has been configured to target this specific file). This inspection will provide the definitive `forward` method signature as understood by the TorchScript runtime, which is paramount for precisely diagnosing why TorchFort continues to expect extraneous arguments. This diagnostic step will clarify whether the root cause lies in the TorchScript export process itself (which is unlikely given the successful Python-side verification) or within TorchFort's model loading and inference infrastructure.

## Reproducibility

To reproduce the current state of the project, ensure all files within this `2025-07-14-a/` directory are present. The environment should be a Singularity container. The sequence of operations to reach this state involves:

1.  **Initial Setup:** Ensure the TorchFort environment is correctly configured and built.
2.  **Python Model Generation & Training:** Execute `burgers1d.py` to train the PINN model, generate `burgers_inference_trained.pt` (the TorchScript inference model), and `burgers1d_python_original_results.bin` (original Python inference results).
3.  **Fortran Training:** Compile and execute `burgers_train.f90` to train the model in Fortran, generating `burgers_model_trained.pt`.
4.  **Python Inference Test:** Execute `inference_test.py` to perform inference using `burgers_inference_trained.pt` and save `burgers1d_python_inference_test_results.bin`.
5.  **Python Comparison:** Execute `compare_python_results.py` to verify the consistency between `burgers1d_python_original_results.bin` and `burgers1d_python_inference_test_results.bin`.

The Fortran inference step (`burgers_inference.f90`) currently fails and is the subject of ongoing debugging.
