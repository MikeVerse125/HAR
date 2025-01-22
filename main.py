import sys
import os 
import torch
from utils.config_util import load_ymal_config
from models.multitask import AttentionModel
from models.baseline import MultiTaskModel
from data.data import DataHandler
from trainer.metrics import Metrics
from trainer.train import Trainer
from eval import TestEvaluator
from utils.plotter import TrainingPlotter


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialise the model based on the configuration 
        if config["model"]["type"] == "attention":
            # Initialise the attention model in this section
            model = AttentionModel(**config["model"]["params"])
        elif config["model"]["type"] == "baseline":
            # Initialise the baseline model in this section
            model = MultiTaskModel(num_classes=config["data"]["num_classes"])
        else:
            raise ValueError(f"Unknown model type: {config['model']['type']}")

        model.to(device)

        # Prepare the data
        data_loaders = DataHandler(
            csv_file=config["data"]["csv_file"],
            img_dir=config["data"]["img_dir"],
            augmentations=config["data"]["augmentation"]
        )
        dataset = data_loaders.get_dataloaders(
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )

        # Initialise the metrics
        metric = Metrics(num_classes=config["data"]["num_classes"], device=device)   

        # Finetune
        finetune = config["model"]["params"]["finetune"]

        # Log file
        log_file = "logs/" + (config["model"]["type"] + "_history.csv" if finetune 
                              else "fine_tune_" + config["model"]["type"] + "_history.csv")
        
        # Model name
        model_name = (config["model"]["type"] if finetune 
                      else "fine_tune_" + config["model"]["type"])
        
        # Train the model
        if config["training"]["status"] == True:
            print("Start training...")   
            train_model = Trainer(model=model, 
                                dataLoaders=dataset, 
                                config=config, 
                                device=device, 
                                metrics=metric, 
                                model_name=model_name + ".pth",
                                log_file=log_file) 
            train_model.fit()

        # Plot the history
        if config["plotter"]["status"] == True:
            print("Start plotting...")
            save_model_path = ("results/ModelsSaved/" + config["model"]["type"]+".pth"if finetune 
                               else"results/ModelsSaved/" + "fine_tune_" + config["model"]["type"]+".pth")
            
            metrics = [metric for metric in config["evaluation"]["metrics"]]
            heads = [head for head in config["evaluation"]["heads"]]

            model.load_state_dict(torch.load(save_model_path))
            plotter = TrainingPlotter(log_file=log_file, model_name=model_name)
            plotter.plot_and_save_metrics(metrics=metrics, heads=heads, save_path="results/metrics")

        # Evaluate the model
        if config["evaluation"]["status"] == True:
            print("Start Evaluating...")
            evaluator = TestEvaluator(
                model=model,                                    # Your trained model
                data_loader=dataset,                            # Test DataLoader
                device=device,
                class_names=data_loaders.dataset.class_mapping, # Map class indices to class names
                log_dir="results/predictions"                   # Directory to save heatmap images
            )
            results = evaluator.evaluate()
            picture_name = ("final_" + config["model"]["type"] + "_heatmap.png" if finetune 
                            else "final_" + "fine_tune_" + config["model"]["type"] + "_heatmap.png")
            evaluator.plot_combined_heatmap(results, picture_name=picture_name)
        

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
