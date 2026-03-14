# CMPE 252 FINAL PROJECT 
# BREAST CANCER DETECTION 
# Group members: Saja Alallao, <add>

# prerequisit:
- Make sure to download the dataset from Kaggle
- Make sure to have folder called dataset->train 
    - Note: split data will split it 80% to train %20 validation

- make sure to install the following: 
    -pip install tensorflow
    -pip install numpy
    -pip install pillow
    -pip install matplotlib
    -pip install scikit-learn
    -pip install tensorflow numpy pillow matplotlib scikit-learn
## Project Structure and Guide 
- `main.py`: The primary for commands.
- `model.py`: Contains model architectures and training logic.
- `split_data.py`: Utility to split raw images into training and validation sets. This will creates additional folder to your
- `visual.py`: Generates accuracy and loss plots after training.
- `config.yaml`: Centralized configuration for hyperparameters and file paths.

# Preper that data
python main.py setup-data

# Train the image Model 
python main.py train-cnn
