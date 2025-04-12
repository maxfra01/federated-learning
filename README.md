# Federated Learning Overview

## What is Federated Learning?
Federated Learning is a machine learning setting where many **clients** (e.g. mobile devices or whole organizations) collaboratively train a model under the orchestration of a central **server** (e.g. service provider), while keeping the training data decentralized. 

## Personal contribution
This repository contains our implementation of a federated learning system using the [CIFAR-100](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html) and [Shakespeare](https://github.com/TalwalkarLab/leaf) datasets. The system includes both centralized and federated training approaches, with support for IID and non-IID data sharding. We have also implemented client selection strategies based on skewed probabilities and a **novel method** based on **entropy of data**.
A detailed report of the project can be found in the `report.pdf` file.

## Repo Structure
- `notebooks/`: Contains Jupyter notebooks to run the experiments.
  
- `src/cifar100/`: Contains the implementation for the CIFAR-100 dataset.
  - `centralized.py`: Centralized training script.
  - `server.py`: Federated learning server script.
  - `client.py`: Client-side training script.
  - `dataset.py`: Dataset handling and sharding.
  - `models.py`: Model definitions.
  - `utils.py`: Utility functions for checkpointing and plotting.
  - `contrib-dataset.py`: Dataset handling and sharding for our novel method based on entropy of data.
  - `contrib-server.py`: Federated learning server script for our novel method based on entropy of data.
- `src/shakespeare/`: Contains the implementation for the Shakespeare dataset.
  - `centralized.py`: Centralized training script.
  - `server.py`: Federated learning server script.
  - `client.py`: Client-side training script.
  - `dataset.py`: Dataset handling and sharding.
  - `model.py`: Model definitions.
  - `utils.py`: Utility functions for checkpointing and plotting.


## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/maxfra01/federated-learning.git
   cd federated-learning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the experiments either using the Jupyter notebooks in the `notebooks/` directory or by running source code directly from the `src/` directory.


## Contributors
- [Massimo Francios](https://github.com/maxfra01)
- [Nicolò Bonincontro](https://github.com/Nick18-beep)
- [Sergio Lampidecchia](https://github.com/sergiolampidecchia)