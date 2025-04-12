import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset


ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def letter_to_index(letter):
    return torch.tensor(ALL_LETTERS.find(letter), dtype=torch.long )

def word_to_indices(word,  n_vocab=NUM_LETTERS):
    '''Returns a list of character indices for a given word'''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

def text_transform(text, max_length=80, vocab_size=NUM_LETTERS):
    '''Transform a string into a tensor with indices instead of one-hot encoding.'''
    # Tokenizzazione: converti ogni lettera in un indice
    indices = [ALL_LETTERS.find(char) for char in text]

    # Padding o Troncamento per lunghezza fissa
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))  # Pad con zeri (carattere vuoto)
    else:
        indices = indices[:max_length]  # Troncamento se il testo è più lungo

    # Restituisci il tensore di indici
    return torch.tensor(indices, dtype=torch.long)

class CentralizedShakespeareDataset(Dataset):
    def __init__(self, root, split, preprocess_params=None, transform=None):
        """
        Args:
            root (str): Path to the dataset directory.
            split (str): Dataset split, either 'train' or 'test'.
            preprocess_params (dict, optional): Parameters for running preprocess.sh script. Keys include:
                - sharding (str): 'iid' or 'niid' for data partitioning.
                - iu (float): Fraction of users if i.i.d. sampling.
                - sf (float): Fraction of data to sample.
                - k (int): Minimum number of samples per user.
                - t (str): 'user' or 'sample' for train-test partition.
                - tf (float): Fraction of data in training set.
                - raw (bool): Include raw text data.
                - smplseed (int): Seed for sampling.
                - spltseed (int): Seed for splitting.
        """
        self.root = root
        self.split = split
        self.preprocess_params = preprocess_params or {}


        # Ensure the working directory is set to the dataset folder
        os.chdir(self.root)

        # Run preprocessing script if needed
        self._preprocess_data()

        # Load the dataset
        self.data = self._load_data()

        # Create a label map to convert string targets to integers
        #self.label_map = self.create_label_map()

    def _preprocess_data(self):
        """Runs preprocess.sh with the given parameters."""
        cmd = "bash preprocess.sh"

        if 'sharding' in self.preprocess_params:
            cmd += f" -s {self.preprocess_params['sharding']}"
        if 'iu' in self.preprocess_params:
            cmd += f" --iu {self.preprocess_params['iu']}"
        if 'sf' in self.preprocess_params:
            cmd += f" --sf {self.preprocess_params['sf']}"
        if 'k' in self.preprocess_params:
            cmd += f" -k {self.preprocess_params['k']}"
        if 't' in self.preprocess_params:
            cmd += f" -t {self.preprocess_params['t']}"
        if 'tf' in self.preprocess_params:
            cmd += f" --tf {self.preprocess_params['tf']}"
        if 'raw' in self.preprocess_params and self.preprocess_params['raw']:
            cmd += f" --raw"
        if 'smplseed' in self.preprocess_params:
            cmd += f" --smplseed {self.preprocess_params['smplseed']}"
        if 'spltseed' in self.preprocess_params:
            cmd += f" --spltseed {self.preprocess_params['spltseed']}"

        print(f"Running command: {cmd}")
        os.system(cmd)
        os.chdir(self.root)

    def _load_data(self):
        """Loads data from the JSON file in the train or test folder, assuming only one file per folder."""
        folder_path = os.path.join(self.root, 'data', self.split)
        print(f"Absolute folder path: {os.path.abspath(folder_path)}")
        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

        if len(json_files) != 1:
            raise ValueError(f"Expected exactly one JSON file in {folder_path}, but found {len(json_files)} files.")

        file_path = os.path.join(folder_path, json_files[0])

        # Carica i dati dal file JSON
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Converti la struttura JSON in un DataFrame di pandas
        records = []
        for user, user_data in data['user_data'].items():
            for x, y in zip(user_data['x'], user_data['y']):
                records.append({
                    'client_id': int(user),
                    'x': x,  # Cambiato input in x
                    'y': y   # Cambiato target in y
                })

        return pd.DataFrame(records)

    def create_label_map(self):
        """Creates a mapping from string labels to integer labels."""
        unique_labels = sorted(self.data['y'].unique())
        print(f"Unique labels: {unique_labels}")  # Debug
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        return label_map

    def get_dataframe(self):
        """Returns the dataset as a pandas DataFrame."""
        return self.data

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'x': self.data.iloc[idx]['x'],
            'y': self.data.iloc[idx]['y']
        }

        sample['x'] = text_transform(sample['x'])  # x is a tensor of one-hot vectors
        sample['y'] = text_transform(sample['y'])  # y is a tensor of one-hot vectors

        return sample['x'], sample['y']
