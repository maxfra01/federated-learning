import os
import torch
import json 

import numpy as np
import matplotlib.pyplot as plt

###################### For Google Colab ######################
# # Mount Google Drive                                     
# from google.colab import drive                         
# drive.mount('/content/drive')
# # Create a Folder in the root directory
# !mkdir -p "/content/drive/My Drive/My Folder/checkpoints"
##############################################################

CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, hyperparameters, subfolder="", data_to_save=None):
    """Save the model checkpoint and remove the previous one."""
    subfolder_path = os.path.join(CHECKPOINT_DIR, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

    # Current and previous file
    filename = f"model_epoch_{epoch}_params_{hyperparameters}.pth"
    filepath = os.path.join(subfolder_path, filename)
    filename_json = f"model_epoch_{epoch}_params_{hyperparameters}.json"
    filepath_json = os.path.join(subfolder_path, filename_json)

    previous_filename = f"model_epoch_{epoch -10}_params_{hyperparameters}.pth"
    previous_filepath = os.path.join(subfolder_path, previous_filename)
    previous_filename_json = f"model_epoch_{epoch -10}_params_{hyperparameters}.json"
    previous_filepath_json = os.path.join(subfolder_path, previous_filename_json)

    # Remove the previous checkpoint
    if epoch > 1 and os.path.exists(previous_filepath) and os.path.exists(previous_filepath_json):
        os.remove(previous_filepath)
        os.remove(previous_filepath_json)

    # Save the new checkpoint
    if optimizer is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
            'epoch': epoch
        }, filepath)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch
        }, filepath)
    print(f"Checkpoint saved: {filepath}")

    with open(filepath_json, 'w') as json_file:
      json.dump(data_to_save, json_file, indent=4)


def load_checkpoint(model, optimizer, hyperparameters, subfolder=""):
    """Load the latest available checkpoint based on hyperparameters."""
    subfolder_path = os.path.join(CHECKPOINT_DIR, subfolder)
    if not os.path.exists(subfolder_path):
        print("No checkpoint found, Starting now...")
        return 1, None  # Epochs start from 1

    # Search for files with the specified hyperparameters
    files = [f for f in os.listdir(subfolder_path) if f"params_{hyperparameters}" in f and f.endswith('.pth')]
    if files:
        # Find the file with the highest epoch
        latest_file = max(files, key=lambda x: int(x.split('_')[2]))
        filepath = os.path.join(subfolder_path, latest_file)
        checkpoint = torch.load(filepath)

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Find and load the associated JSON file
        json_filename = latest_file.replace('.pth', '.json')
        json_filepath = os.path.join(subfolder_path, json_filename)
        json_data = None
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r') as json_file:
                json_data = json.load(json_file)
            print(f"JSON data loaded: {json_filepath}")
        else:
            print(f"No JSON file found for: {latest_file}")

        print(f"Checkpoint found: Resume epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1, json_data

    print("No checkpoint found, Starting now...")
    return 1, None  # Epochs start from 1


def generate_skewed_probabilities(num_clients, gamma):
    """It generates skewed probabilities for clients using a Dirichlet distribution."""
    probabilities = np.random.dirichlet([gamma] * num_clients)
    return probabilities

def plot_selected_clients_distribution(selected_clients_per_round, num_clients, hyperparameters):
    """Plot the distribution of selected clients at the end of the process."""
    counts = np.zeros(num_clients)

    # Count how many times each client was selected in all rounds
    for selected_clients in selected_clients_per_round:
        for client in selected_clients:
            counts[client] += 1

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clients), counts, color='skyblue', edgecolor='black')
    plt.title("Distribution of Selected Clients During Federated Averaging")
    plt.xlabel("Client ID")
    plt.ylabel("Selection Frequency")
    plt.grid(axis='y')
    plt.savefig(f"CIFAR100_Client_distribution_{hyperparameters}.png")
    plt.show()