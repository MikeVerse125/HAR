import yaml
import os

def load_ymal_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file} is not found")

    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
    return config