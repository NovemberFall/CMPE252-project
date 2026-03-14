
import os
# Silence TensorFlow logs BEFORE importing anything else
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import sys
import yaml
from model import train_tabular, train_cnn, MODEL_PATH
from splitData import splitDataset  # Import the function you just wrote
from pathlib import Path

#loading the config file
def load_config():
    config_path = Path("config.yaml") # calling the config file path
    if not config_path.exists():
        print("Error: config.yaml not found!") #throwing error if the path doesn't exist
        exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f) # is the file exsit proceed to load it 
    
def main():
    parser = argparse.ArgumentParser(description="Breast Cancer Classification") #This handles the overall help message
    subparsers = parser.add_subparsers(dest="mode", required=True) #To ensures the program crashes with a helpful error if the user forgets to pick a mode

    # Train Tabular
    t_tab = subparsers.add_parser("train-tab", help="Train RF on tabular data")
    t_tab.add_argument("estimators", type=int, default=100)

    # Train CNN
    #Sends the tabular file paths and number of trees to estimators
    t_cnn = subparsers.add_parser("train-cnn", help="Train CNN on BUSI images")
    t_cnn.add_argument("--train-dir", required=True)
    t_cnn.add_argument("--val-dir", required=True)
    t_cnn.add_argument("--epochs", type=int, default=5)

    config = load_config()
    
    parser = argparse.ArgumentParser(description="Breast Cancer Classification Toolkit")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Command: setup-data
    subparsers.add_parser("setup-data", help="Split images into train/val folders")

    # Command: train-tab
    subparsers.add_parser("train-tab", help="Train RF on tabular data")

    # Command: train-cnn
    t_cnn = subparsers.add_parser("train-cnn", help="Train CNN on BUSI images")
    t_cnn.add_argument("--epochs", type=int, help="Override config epochs")

    args = parser.parse_args()

    if args.mode == "setup-data":
        # Uses ratio from config
        splitDataset(ratio=config['data']['split_ratio'])

    elif args.mode == "train-tab":
        train_tabular(
            save_path=Path(config['tabular']['model_save_path']),
            n_estimators=config['tabular']['n_estimators']
        )

    elif args.mode == "train-cnn":
        # Use argparse override if provided, otherwise use config
        epochs = args.epochs or config['cnn']['epochs']
        
        train_cnn(
            train_dir=config['data']['train_dir'],
            val_dir=config['data']['val_dir'],
            epochs=epochs,
            batch_size=config['cnn']['batch_size']
        )

if __name__ == "__main__":
    main()