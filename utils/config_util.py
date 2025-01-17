import yaml
import os

def load_ymal_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file} is not found")

    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            # Force type conversion for critical fields
            config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
            config["training"]["classification_weight"] = float(config["training"].get("classification_weight", 1.0))
            config["training"]["binary_weight"] = float(config["training"].get("binary_weight", 1.0))
            config["training"]["epochs"] = int(config["training"]["epochs"])
            config["training"]["patience"] = int(config["training"]["patience"])
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
    return config