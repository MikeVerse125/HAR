import sys
import os 
import torch
from utils.config_util import load_ymal_config
from A1_2024_data.data import get_dataLoaders

def main():
    try: 
        if len(sys.argv) != 2:
            raise ValueError("Usage: python main.py <config.yaml>")
            
        # Get the config file path
        config_file = "cfg/" + sys.argv[1]

        # Ensure the config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' is not found")
        
        print(f"Using the config file: {config_file}")

        # Load YAML configuration file
        try:
            config = load_ymal_config(config_file)
        except ValueError as e:
            raise ValueError(f"Error loading the config file: {e}")

        #set device (GPU or CPU)
        device = torch.device("cude" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialise the model based on the configuration 
        if config["model"]["type"] == "attenttion":
            # Initialise the attention model in this section
            
        elif config["model"]["type"] == "baseline"
            # Initialise the baseline model in this section
        else:
            raise ValueError(f"Unknown model type: {config['model']['type']}")

        # Prepare the data

            

        
    except ValueError as e:
        print(f"ValueError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FileNotFoundError: {e}")
        sys.exit(1)
    except KeyError as e: 
        print(f"KeyError: {e}")
        sys.exit(1)
    except RuntimeError as e: 
        print(f"RuntimeError: {e}")
        sys.exit(1)
    except IOError as e: 
        print(f"IOError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
