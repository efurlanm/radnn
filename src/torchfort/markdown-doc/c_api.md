---
title: TorchFort C API
---

These are all the types and functions available in the TorchFort C API.

General
=======

Types
-----

### torchfort\_datatype\_t {#torchfort_datatype_t-ref}

------------------------------------------------------------------------

### torchfort\_result\_t {#torchfort_result_t-ref}

------------------------------------------------------------------------

### torchfort\_tensor\_list\_t {#torchfort_tensor_list_t-ref}

------------------------------------------------------------------------

Global Context Settings
-----------------------

These are global routines which affect the behavior of the libtorch
backend. It is therefore recommended to call these functions before any
other TorchFort calls are made.

### torchfort\_set\_cudnn\_benchmark {#torchfort_set_cudnn_benchmark-ref}

::: {.doxygenfunction}
torchfort\_set\_cudnn\_benchmark
:::

Tensor List Management
----------------------

### torchfort\_tensor\_list\_create {#torchfort_tensor_list_create-ref}

::: {.doxygenfunction}
torchfort\_tensor\_list\_create
:::

------------------------------------------------------------------------

### torchfort\_tensor\_list\_destroy {#torchfort_tensor_list_destroy-ref}

::: {.doxygenfunction}
torchfort\_tensor\_list\_destroy
:::

------------------------------------------------------------------------

### torchfort\_tensor\_list\_add\_tensor {#torchfort_tensor_list_add_tensor-ref}

::: {.doxygenfunction}
torchfort\_tensor\_list\_add\_tensor
:::

------------------------------------------------------------------------

Supervised Learning {#torchfort_general_c-ref}
===================

Model Creation
--------------

### torchfort\_create\_model {#torchfort_create_model-ref}

::: {.doxygenfunction}
torchfort\_create\_model
:::

------------------------------------------------------------------------

### torchfort\_create\_distributed\_model {#torchfort_create_distributed-model-ref}

::: {.doxygenfunction}
torchfort\_create\_distributed\_model
:::

------------------------------------------------------------------------

Model Training/Inference
------------------------

### torchfort\_train {#torchfort_train-ref}

::: {.doxygenfunction}
torchfort\_train
:::

------------------------------------------------------------------------

### torchfort\_train\_multiarg {#torchfort_train_multiarg-ref}

::: {.doxygenfunction}
torchfort\_train\_multiarg
:::

------------------------------------------------------------------------

### torchfort\_inference {#torchfort_inference-ref}

::: {.doxygenfunction}
torchfort\_inference
:::

------------------------------------------------------------------------

### torchfort\_inference\_multiarg {#torchfort_inference_multiarg-ref}

::: {.doxygenfunction}
torchfort\_inference\_multiarg
:::

------------------------------------------------------------------------

Model Management
----------------

### torchfort\_save\_model {#torchfort_save_model-ref}

::: {.doxygenfunction}
torchfort\_save\_model
:::

------------------------------------------------------------------------

### torchfort\_load\_model {#torchfort_load_model-ref}

::: {.doxygenfunction}
torchfort\_load\_model
:::

------------------------------------------------------------------------

### torchfort\_save\_checkpoint {#torchfort_save_checkpoint-ref}

::: {.doxygenfunction}
torchfort\_save\_checkpoint
:::

------------------------------------------------------------------------

### torchfort\_load\_checkpoint {#torchfort_load_checkpoint-ref}

::: {.doxygenfunction}
torchfort\_load\_checkpoint
:::

------------------------------------------------------------------------

Weights and Biases Logging
--------------------------

### torchfort\_wandb\_log\_int {#torchfort_wandb_log_int-ref}

::: {.doxygenfunction}
torchfort\_wandb\_log\_int
:::

------------------------------------------------------------------------

### torchfort\_wandb\_log\_float {#torchfort_wandb_log_float-ref}

::: {.doxygenfunction}
torchfort\_wandb\_log\_float
:::

------------------------------------------------------------------------

### torchfort\_wandb\_log\_double {#torchfort_wandb_log_double-ref}

::: {.doxygenfunction}
torchfort\_wandb\_log\_double
:::

------------------------------------------------------------------------

Reinforcement Learning {#torchfort_rl_c-ref}
======================

Similar to other reinforcement learning frameworks such as [Spinning
Up](https://spinningup.openai.com/en/latest/) from OpenAI or [Stable
Baselines](https://stable-baselines3.readthedocs.io/en/master/), we
distinguish between on-policy and off-policy algorithms since those two
types require different APIs.

------------------------------------------------------------------------

Off-Policy Algorithms {#torchfort_rl_off_policy_c-ref}
---------------------

### System Creation

Basic routines to create and register a reinforcement learning system in
the internal registry. A (synchronous) data parallel distributed option
is available.

#### torchfort\_rl\_off\_policy\_create\_system {#torchfort_rl_off_policy_create_system-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_create\_system
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_create\_distributed\_system {#torchfort_rl_off_policy_create_distributed_system-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_create\_distributed\_system
:::

------------------------------------------------------------------------

### Training/Evaluation

These routines are used for training the reinforcement learning system
or for steering the environment.

#### torchfort\_rl\_off\_policy\_train\_step {#torchfort_rl_off_policy_train_step-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_train\_step
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_predict\_explore {#torchfort_rl_off_policy_predict_explore-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_predict\_explore
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_predict {#torchfort_rl_off_policy_predict-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_predict
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_evaluate {#torchfort_rl_off_policy_evaluate-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_evaluate
:::

------------------------------------------------------------------------

### System Management

The purpose of these routines is to manage the reinforcement learning
systems internal data. It allows the user to add tuples to the replay
buffer and query the system for readiness. Additionally, save and
restore functionality is also provided.

#### torchfort\_rl\_off\_policy\_update\_replay\_buffer {#torchfort_rl_off_policy_update_replay_buffer-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_update\_replay\_buffer
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_update\_replay\_buffer\_multi {#torchfort_rl_off_policy_update_replay_buffer_multi-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_update\_replay\_buffer\_multi
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_is\_ready {#torchfort_rl_off_policy_is_ready-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_is\_ready
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_save\_checkpoint {#torchfort_rl_off_policy_save_checkpoint-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_save\_checkpoint
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_load\_checkpoint {#torchfort_rl_off_policy_load_checkpoint-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_load\_checkpoint
:::

------------------------------------------------------------------------

### Weights and Biases Logging

The reinforcement learning system performs logging for all involved
networks automatically during training. The following routines are
provided for additional logging of system relevant quantities, such as
e.g. the accumulated reward.

#### torchfort\_rl\_off\_policy\_wandb\_log\_int {#torchfort_rl_off_policy_wandb_log_int-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_wandb\_log\_int
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_wandb\_log\_float {#torchfort_rl_off_policy_wandb_log_float-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_wandb\_log\_float
:::

------------------------------------------------------------------------

#### torchfort\_rl\_off\_policy\_wandb\_log\_double {#torchfort_rl_off_policy_wandb_log_double-ref}

::: {.doxygenfunction}
torchfort\_rl\_off\_policy\_wandb\_log\_double
:::

------------------------------------------------------------------------

On-Policy Algorithms {#torchfort_rl_on_policy_c-ref}
--------------------

### System Creation

Basic routines to create and register a reinforcement learning system in
the internal registry. A (synchronous) data parallel distributed option
is available.

#### torchfort\_rl\_on\_policy\_create\_system {#torchfort_rl_on_policy_create_system-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_create\_system
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_create\_distributed\_system {#torchfort_rl_on_policy_create_distributed_system-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_create\_distributed\_system
:::

------------------------------------------------------------------------

### Training/Evaluation

These routines are used for training the reinforcement learning system
or for steering the environment.

#### torchfort\_rl\_on\_policy\_train\_step {#torchfort_rl_on_policy_train_step-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_train\_step
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_predict\_explore {#torchfort_rl_on_policy_predict_explore-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_predict\_explore
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_predict {#torchfort_rl_on_policy_predict-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_predict
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_evaluate {#torchfort_rl_on_policy_evaluate-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_evaluate
:::

------------------------------------------------------------------------

### System Management

The purpose of these routines is to manage the reinforcement learning
systems internal data. It allows the user to add tuples to the replay
buffer and query the system for readiness. Additionally, save and
restore functionality is also provided.

#### torchfort\_rl\_on\_policy\_update\_rollout\_buffer {#torchfort_rl_on_policy_update_rollout_buffer-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_update\_rollout\_buffer
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_update\_rollout\_buffer\_multi {#torchfort_rl_on_policy_update_rollout_buffer_multi-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_update\_rollout\_buffer\_multi
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_is\_ready {#torchfort_rl_on_policy_is_ready-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_is\_ready
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_save\_checkpoint {#torchfort_rl_on_policy_save_checkpoint-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_save\_checkpoint
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_load\_checkpoint {#torchfort_rl_on_policy_load_checkpoint-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_load\_checkpoint
:::

------------------------------------------------------------------------

### Weights and Biases Logging

The reinforcement learning system performs logging for all involved
networks automatically during training. The following routines are
provided for additional logging of system relevant quantities, such as
e.g. the accumulated reward.

#### torchfort\_rl\_on\_policy\_wandb\_log\_int {#torchfort_rl_on_policy_wandb_log_int-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_wandb\_log\_int
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_wandb\_log\_float {#torchfort_rl_on_policy_wandb_log_float-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_wandb\_log\_float
:::

------------------------------------------------------------------------

#### torchfort\_rl\_on\_policy\_wandb\_log\_double {#torchfort_rl_on_policy_wandb_log_double-ref}

::: {.doxygenfunction}
torchfort\_rl\_on\_policy\_wandb\_log\_double
:::

<br><sub>Last edited: 2025-04-29 15:27:17</sub>
