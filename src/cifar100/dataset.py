import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR100Dataset(Dataset):
    def __init__(self, root, split='train', transform=None, sharding=None, K=10, Nc=2):
        """
        CIFAR-100 Dataset with IID and non-IID sharding.

        Args:
        - root (str): Directory to store the dataset.
        - split (str): 'train' or 'test'.
        - transform (callable): Transformations applied to the images.
        - sharding (str): 'iid' or 'niid'.
        - K (int): Number of clients for the sharding.
        - Nc (int): Number of classes per client (used for non-iid sharding).
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.sharding = sharding
        self.K = K
        self.Nc = Nc


        # Default transformations if none are provided
        if self.transform is None:
            if self.split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),  # Flip orizzontale casuale
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),  # Converte l'immagine in un tensore PyTorch
                    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),  # Normalizzazione
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),  # Converte in tensore PyTorch
                    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),  # Normalizzazione
                ])

        dataset = datasets.CIFAR100(
            root=self.root,
            train=(self.split == 'train'),
            download=True
        )

        self.data = pd.DataFrame({
            "image": [dataset[i][0] for i in range(len(dataset))],
            "label": [dataset[i][1] for i in range(len(dataset))]
        })

        if self.split == 'train' and self.sharding:
            self.data = self._apply_sharding()

    def _apply_sharding(self):
        """Apply IID or non-IID sharding to the training data."""
        if self.sharding == 'iid':
            return self._iid_sharding()
        elif self.sharding == 'niid':
            return self._non_iid_sharding()
        else:
            raise ValueError("Sharding must be 'iid' or 'niid'.")

    def _iid_sharding(self):
        """Split data IID: uniformly distribute samples across K clients."""
        data_split = []
        indices = self.data.index.tolist()
        random.shuffle(indices)

        # Split indices equally among K clients
        client_indices = [indices[i::self.K] for i in range(self.K)]

        for client_id, idxs in enumerate(client_indices):
            client_data = self.data.loc[idxs].copy()
            client_data['client_id'] = client_id
            data_split.append(client_data)

        return pd.concat(data_split, ignore_index=True)

    def _non_iid_sharding(self):
        """Non-IID sharding with fixed number of classes per client"""
        data_split = []
        labels = self.data['label'].unique()
        samples_per_client = len(self.data) // self.K

        for client_id in range(self.K):
            # Seleziona Nc classi casuali per questo client
            client_classes = np.random.choice(labels, size=self.Nc, replace=False)

            # Ottieni i dati per le classi selezionate
            client_data = pd.DataFrame()
            samples_per_class = samples_per_client // self.Nc

            #print(f"Client {client_id}:")
            #print(f"Classi assegnate: {client_classes}")

            for idx, class_ in enumerate(client_classes):
                class_data = self.data[self.data['label'] == class_]
                samples = class_data.sample(n=samples_per_class, replace=True)
                client_data = pd.concat([client_data, samples])

                #print(f"  Classe {class_}: {len(samples)} campioni")

            client_data['client_id'] = client_id
            data_split.append(client_data)

        # Concatenate tutti i dati per avere il dataset completo per tutti i clienti
        all_data = pd.concat(data_split, ignore_index=True)
        return all_data

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image, label = row['image'], row['label']

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)