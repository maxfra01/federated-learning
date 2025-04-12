import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils import save_checkpoint, load_checkpoint, plot_selected_clients_distribution, generate_skewed_probabilities
from centralized import evaluate_model
from dataset import CIFAR100Dataset
from models import LeNet5
from client import Client

class Server:

  def __init__(self, model, clients, test_data):
    self.model = model
    self.clients = clients
    self.test_data = test_data
    self.round_losses = []
    self.round_accuracies = []
    self.selected_clients_per_round = [] # clients selected for skewness

  def federated_averaging(self, local_steps, batch_size, num_rounds, fraction_fit, skewness = None, hyperparameters = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)
     # Load checkpoint if it exists
    data_to_load = None
    if skewness is  None:
      start_epoch, data_to_load = load_checkpoint(self.model,optimizer=None,hyperparameters=hyperparameters, subfolder="Federated_Uniform/")
    else:
      start_epoch, data_to_load = load_checkpoint(self.model,optimizer=None,hyperparameters=hyperparameters, subfolder="Federated_Skewed/")

    if data_to_load is not None:
      self.round_losses = data_to_load['round_losses']
      self.round_accuracies = data_to_load['round_accuracies']
      self.selected_clients_per_round = data_to_load['selected_clients_per_round']


    for round in range(start_epoch, num_rounds+1):

      if skewness is not None:
        probabilities = generate_skewed_probabilities(len(self.clients), skewness)
        selected_clients = np.random.choice(self.clients, size=max(1, int(fraction_fit*len(self.clients))), replace=False, p=probabilities)

      else:
        selected_clients = np.random.choice(self.clients, size=max(1, int(fraction_fit*len(self.clients))), replace=False)

      self.selected_clients_per_round.append([client.client_id for client in selected_clients])


      global_weights = self.model.state_dict()

      # Simulating parallel clients training
      client_weights = {}
      for client in selected_clients:
        client_weights[client.client_id] = client.train(global_weights, local_steps, batch_size)

      new_global_weights = {key: torch.zeros_like(value).type(torch.float32) for key, value in global_weights.items()}

      total_data_size = sum([len(client.data) for client in selected_clients])
      for client in selected_clients:
        scaling_factor = len(client.data) / total_data_size
        for key in new_global_weights.keys():
          new_global_weights[key] += scaling_factor * client_weights[client.client_id][key]

      # Update global model weights
      self.model.load_state_dict(new_global_weights)

      # Evaluate global model every 10 rounds
      if round % 10 == 0:
        loss, accuracy = evaluate_model(self.model, DataLoader(self.test_data, batch_size=batch_size, shuffle=True, pin_memory=True), nn.CrossEntropyLoss(), device)
        self.round_losses.append(loss)
        self.round_accuracies.append(accuracy)
        print(f"Round {round}/{num_rounds} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        data_to_save = {
          'round_losses': self.round_losses,
          'round_accuracies': self.round_accuracies,
          'selected_clients_per_round': [[client for client in round_clients] for round_clients in self.selected_clients_per_round]  # Serialize only client_ids
      }

        if skewness is  None:
          save_checkpoint(self.model, None, round , hyperparameters, "Federated_Uniform/", data_to_save)
        else:
          save_checkpoint(self.model, None, round , hyperparameters, "Federated_Skewed/", data_to_save)

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(self.round_losses, label='CIFAR-100 Test Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(self.round_accuracies, label='CIFAR-100 Test Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    if skewness is  None:
      plt.savefig(f"CIFAR100_fedavg_uniform{hyperparameters}.jpg")
    else:
      plt.savefig(f"CIFAR100_fedavg_skew{hyperparameters}.jpg")

    plt.show()

    plot_selected_clients_distribution(self.selected_clients_per_round, len(self.clients), hyperparameters)


if __name__ == '__main__':
  
  DIR_DATA = "./data"
  
  K = 100 # fixed
  LOCAL_STEPS = 4 # J
  ROUNDS = 2000
  C = 0.1 # fixed
  BATCH_SIZE = 64
  LR = 0.01
  MOMENTUM = 0.9
  WEIGHT_DECAY = 1e-4
  SKEWNESS=0.01

  optimizer_params = {
      "lr": LR,
      "momentum": MOMENTUM,
      "weight_decay": WEIGHT_DECAY
  }

  model_cifar = LeNet5(100)

  train_dataset = CIFAR100Dataset(DIR_DATA, split='train', sharding='iid', K=K)
  test_dataset = CIFAR100Dataset(DIR_DATA, split='test')

  clients = []
  for i in range(K):
    client_data = Subset(train_dataset, train_dataset.data[train_dataset.data["client_id"] == i].index)
    clients.append(Client(model_cifar, i, client_data, optimizer_params))


  server_uniform = Server(model_cifar, clients, test_dataset)
  hyperparameters = f"BS{BATCH_SIZE}_LR{LR}_M{MOMENTUM}_WD{WEIGHT_DECAY}_J{LOCAL_STEPS}_C{C}"
  server_uniform.federated_averaging(local_steps=LOCAL_STEPS, batch_size=BATCH_SIZE, num_rounds=ROUNDS, fraction_fit=C,hyperparameters=hyperparameters)



  #server_skewed = Server(model_cifar, clients, test_dataset)
  #hyperparameters = f"BS{BATCH_SIZE}_LR{LR}_M{MOMENTUM}_WD{WEIGHT_DECAY}_J{LOCAL_STEPS}_C{C}_SK{SKEWNESS}"
  #server_skewed.federated_averaging(local_steps=LOCAL_STEPS, batch_size=BATCH_SIZE, num_rounds=ROUNDS, fraction_fit=C, skewness=SKEWNESS, hyperparameters=hyperparameters)
