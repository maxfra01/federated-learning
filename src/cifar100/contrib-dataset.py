import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def generate_skewed_array(target_sum, num_elements, skew_factor=0):
    """
    Generate an array of integers that sum up to a target value.

    Args:
        target_sum (int): The target sum of the array.
        num_elements (int): The number of elements in the array.
        skew_factor (float): A positive skew factor favors smaller numbers,
                             a negative skew factor favors larger numbers,
                             and 0 results in an approximately uniform distribution.

    Returns:
        list: An array of integers summing up to the target value.
    """
    if num_elements <= 0:
        raise ValueError("num_elements must be greater than 0")

    # Generate a distribution using Dirichlet
    alpha = [1 + skew_factor] * num_elements if skew_factor >= 0 else [1 - skew_factor] * num_elements
    raw_weights = np.random.dirichlet(alpha)

    # Scale weights to the target sum
    scaled_weights = raw_weights * target_sum

    # Convert to integers while keeping track of the fractional parts
    integer_parts = np.floor(scaled_weights).astype(int)
    fractional_parts = scaled_weights - integer_parts

    # Adjust the result to ensure the sum matches the target_sum
    diff = target_sum - np.sum(integer_parts)

    # Distribute the difference based on the largest fractional parts
    fractional_indices = np.argsort(-fractional_parts)
    for i in range(abs(diff)):
        integer_parts[fractional_indices[i]] += 1 if diff > 0 else -1

    return integer_parts.tolist()

def generate_vector(target_sum, n):
    """
    Generates a vector (list) with n positive elements that sum to target_sum.
    Used for niid sharding

    Parameters:
    - target_sum: The desired sum of the vector elements.
    - n: The number of elements in the vector.

    Returns:
    - A list of n positive elements that sum to target_sum.
    """
    if target_sum < n:
        raise ValueError("Target sum must be at least equal to the number of elements to ensure all elements are positive.")

    # Create n random weights that sum to 1
    weights = [random.random() for _ in range(n)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Scale the weights to sum to the target_sum
    vector = [max(1, int(target_sum * w)) for w in normalized_weights]

    # Adjust to ensure the exact sum of target_sum
    current_sum = sum(vector)
    while current_sum != target_sum:
        # Find the difference between the current sum and the target sum
        difference = target_sum - current_sum

        # Adjust a random element to fix the difference
        index_to_adjust = random.randint(0, n - 1)
        if difference > 0:
            vector[index_to_adjust] += 1
        elif vector[index_to_adjust] > 1:  # Ensure no element goes below 1
            vector[index_to_adjust] -= 1

        # Recalculate the sum
        current_sum = sum(vector)

    return vector


class CIFAR100DatasetContribution(Dataset):
    def __init__(self, root, split='train', transform=None, sharding=None, K=10, Nc=2, skewed_factor = None):
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
      samples_per_client = generate_skewed_array(len(self.data), self.K)

      for client_id in range(self.K):
          # Seleziona Nc classi casuali per questo client
          client_classes = np.random.choice(labels, size=self.Nc, replace=False)

          # Ottieni il numero totale di campioni per il client
          total_samples = samples_per_client[client_id]

          # Assicurati che il numero totale di campioni sia sufficiente per il numero di classi
          min_samples_per_class = 5  # Numero minimo di campioni per classe
          total_samples = max(total_samples, min_samples_per_class * len(client_classes))

          # Distribuisci i campioni tra le classi
          samples_per_class = generate_vector(total_samples, len(client_classes))

          client_data = pd.DataFrame()
          for idx, class_ in enumerate(client_classes):
              class_data = self.data[self.data['label'] == class_]
              samples = class_data.sample(n=samples_per_class[idx], replace=True)  # replace=False per evitare duplicati
              client_data = pd.concat([client_data, samples])

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
