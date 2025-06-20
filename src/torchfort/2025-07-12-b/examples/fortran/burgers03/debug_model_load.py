import torch
import yaml

model_path = "burgers_inference_trained.pt"

try:
    # Try loading the model
    model = torch.jit.load(model_path)
    print(f"Successfully loaded model from {model_path}")
    print("Model structure:")
    print(model)
except Exception as e:
    print(f"Error loading model {model_path}: {e}")

try:
    # Try loading the YAML configuration
    config_path = "burgers_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Successfully loaded YAML configuration from {config_path}")
    print("YAML content:")
    print(config)
except Exception as e:
    print(f"Error loading YAML configuration {config_path}: {e}")