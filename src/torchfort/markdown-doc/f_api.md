---
title: TorchFort Fortran API
---

These are all the types and functions available in the TorchFort Fortran
API.

General
=======

Types
-----

### torchfort\_datatype {#torchfort_datatype_t-f-ref}

See documentation for equivalent C enumerator,
`torchfort_datatype_t-ref`{.interpreted-text role="ref"}.

------------------------------------------------------------------------

### torchfort\_result {#torchfort_result_t-f-ref}

See documentation for equivalent C enumerator,
`torchfort_result_t-ref`{.interpreted-text role="ref"}.

------------------------------------------------------------------------

### torchfort\_tensor\_list {#torchfort_tensor_list_t-f-ref}

See documentation for equivalent C typedef,
`torchfort_tensor_list_t-ref`{.interpreted-text role="ref"}.

------------------------------------------------------------------------

Global Context Settings
-----------------------

These are global routines which affect the behavior of the libtorch
backend. It is therefore recommended to call these functions before any
other TorchFort calls are made.

### torchfort\_set\_cudnn\_benchmark {#torchfort_set_cudnn_benchmark-f-ref}

------------------------------------------------------------------------

Tensor List Management
----------------------

### torchfort\_tensor\_list\_create {#torchfort_tensor_list_create-f-ref}

------------------------------------------------------------------------

### torchfort\_tensor\_list\_destroy {#torchfort_tensor_list_destroy-f-ref}

------------------------------------------------------------------------

### torchfort\_tensor\_list\_add\_tensor {#torchfort_tensor_list_add_tensor-f-ref}

------------------------------------------------------------------------

Supervised Learning {#torchfort_general_f-ref}
===================

Model Creation
--------------

### torchfort\_create\_model {#torchfort_create_model-f-ref}

------------------------------------------------------------------------

### torchfort\_create\_distributed\_model {#torchfort_create_distributed_model-f-ref}

------------------------------------------------------------------------

Model Training/Inference
------------------------

### torchfort\_train {#torchfort_train-f-ref}

------------------------------------------------------------------------

### torchfort\_train\_multiarg {#torchfort_train_multiarg-f-ref}

------------------------------------------------------------------------

### torchfort\_inference {#torchfort_inference-f-ref}

------------------------------------------------------------------------

### torchfort\_inference\_multiarg {#torchfort_inference_multiarg-f-ref}

------------------------------------------------------------------------

Model Management
----------------

### torchfort\_save\_model {#torchfort_save_model-f-ref}

------------------------------------------------------------------------

### torchfort\_load\_model {#torchfort_load_model-f-ref}

------------------------------------------------------------------------

### torchfort\_save\_checkpoint {#torchfort_save_checkpoint-f-ref}

------------------------------------------------------------------------

### torchfort\_load\_checkpoint {#torchfort_load_checkpoint-f-ref}

------------------------------------------------------------------------

Weights and Biases Logging
--------------------------

### torchfort\_wandb\_log\_int {#torchfort_wandb_log_int-f-ref}

------------------------------------------------------------------------

### torchfort\_wandb\_log\_float {#torchfort_wandb_log_float-f-ref}

------------------------------------------------------------------------

### torchfort\_wandb\_log\_double {#torchfort_wandb_log_double-f-ref}

------------------------------------------------------------------------

Reinforcement Learning {#torchfort_rl_f-ref}
======================

Similar to other reinforcement learning frameworks such as [Spinning
Up](https://spinningup.openai.com/en/latest/) from OpenAI or [Stable
Baselines](https://stable-baselines3.readthedocs.io/en/master/), we
distinguish between on-policy and off-policy algorithms since those two
types require different APIs.

------------------------------------------------------------------------

Off-Policy Algorithms {#torchfort_rl_off_policy_f-ref}
---------------------

### System Creation

Basic routines to create and register a reinforcement learning system in
the internal registry. A (synchronous) data parallel distributed option
is available.

#### torchfort\_rl\_off\_policy\_create\_system {#torchfort_rl_off_policy_create_system-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_create\_distributed\_system {#torchfort_rl_off_policy_create_distributed_system-f-ref}

------------------------------------------------------------------------

### Training/Evaluation

These routines are be used for training the reinforcement learning
system or for steering the environment.

#### torchfort\_rl\_off\_policy\_train\_step {#torchfort_rl_off_policy_train_step-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_predict\_explore {#torchfort_rl_off_policy_predict_explore-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_predict {#torchfort_rl_off_policy_predict-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_evaluate {#torchfort_rl_off_policy_evaluate-f-ref}

------------------------------------------------------------------------

### System Management

#### torchfort\_rl\_off\_policy\_update\_replay\_buffer {#torchfort_rl_off_policy_update_replay_buffer-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_is\_ready {#torchfort_rl_off_policy_is_ready-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_save\_checkpoint {#torchfort_rl_off_policy_save_checkpoint-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_load\_checkpoint {#torchfort_rl_off_policy_load_checkpoint-f-ref}

------------------------------------------------------------------------

### Weights and Biases Logging

#### torchfort\_rl\_off\_policy\_wandb\_log\_int {#torchfort_rl_off_policy_wandb_log_int-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_wandb\_log\_float {#torchfort_rl_off_policy_wandb_log_float-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_wandb\_log\_double {#torchfort_rl_off_policy_wandb_log_double-f-ref}

------------------------------------------------------------------------

On-Policy Algorithms {#torchfort_rl_on_policy_f-ref}
--------------------

### System Creation

Basic routines to create and register a reinforcement learning system in
the internal registry. A (synchronous) data parallel distributed option
is available.

#### torchfort\_rl\_on\_policy\_create\_system {#torchfort_rl_on_policy_create_system-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_create\_distributed\_system {#torchfort_rl_on_policy_create_distributed_system-f-ref}

------------------------------------------------------------------------

### Training/Evaluation

These routines are be used for training the reinforcement learning
system or for steering the environment.

#### torchfort\_rl\_on\_policy\_train\_step {#torchfort_rl_on_policy_train_step-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_predict\_explore {#torchfort_rl_on_policy_predict_explore-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_predict {#torchfort_rl_on_policy_predict-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_evaluate {#torchfort_rl_on_policy_evaluate-f-ref}

------------------------------------------------------------------------

### System Management

#### torchfort\_rl\_on\_policy\_update\_rollout\_buffer {#torchfort_rl_on_policy_update_rollout_buffer-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_reset\_rollout\_buffer {#torchfort_rl_on_policy_reset_rollout_buffer-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_is\_ready {#torchfort_rl_on_policy_is_ready-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_save\_checkpoint {#torchfort_rl_on_policy_save_checkpoint-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_load\_checkpoint {#torchfort_rl_on_policy_load_checkpoint-f-ref}

------------------------------------------------------------------------

### Weights and Biases Logging

#### torchfort\_rl\_on\_policy\_wandb\_log\_int {#torchfort_rl_on_policy_wandb_log_int-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_wandb\_log\_float {#torchfort_rl_on_policy_wandb_log_float-f-ref}

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_wandb\_log\_double {#torchfort_rl_on_policy_wandb_log_double-f-ref}

<br><sub>Last edited: 2025-04-29 15:27:53</sub>
