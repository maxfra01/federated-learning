import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils import save_checkpoint, load_checkpoint, plot_selected_clients_distribution
from centralized import evaluate_model

def federated_averaging(self, local_steps, batch_size, num_rounds, fraction_fit, alpha = None, hyperparameters = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)

     # Carica il checkpoint se esiste
    data_to_load = None
    probabilities = None

    start_epoch, data_to_load = load_checkpoint(self.model,optimizer=None,hyperparameters=hyperparameters, subfolder="Federated_Uniform/")

    if data_to_load is not None:
      self.round_losses = data_to_load['round_losses']
      self.round_accuracies = data_to_load['round_accuracies']
      self.selected_clients_per_round = data_to_load['selected_clients_per_round']

    probabilities=self.evaluate_clients(alpha) # Generate data/entropy prob

    for round in range(start_epoch, num_rounds+1):

      selected_clients = self.client_selection(len(self.clients), fraction_fit, probabilities)

      self.selected_clients_per_round.append([self.clients[client_idx].client_id for client_idx in selected_clients])

      global_weights = self.model.state_dict()

      # Simulating parallel clients training
      client_weights = {}
      for client_idx in selected_clients:
        client = self.clients[client_idx]  # Accedi all'oggetto Client usando l'indice
        client_weights[client.client_id] = client.train(global_weights, local_steps, batch_size)


      new_global_weights = {key: torch.zeros_like(value).type(torch.float32) for key, value in global_weights.items()}

      total_data_size = sum([len(self.clients[client_idx].data) for client_idx in selected_clients])
      for client_idx in selected_clients:
          client = self.clients[client_idx]  # Accedi all'oggetto Client
          scaling_factor = len(client.data) / total_data_size
          for key in new_global_weights.keys():
              new_global_weights[key] += scaling_factor * client_weights[client.client_id][key]

      # Update global model weights
      self.model.load_state_dict(new_global_weights)

      # Evaluate global model every 10 rounds
      if round % 10 == 0:
        loss, accuracy = evaluate_model(self.model, DataLoader(self.val_data, batch_size=batch_size, shuffle=False, pin_memory=True), nn.CrossEntropyLoss(), device)
        loss_test, accuracy_test = evaluate_model(self.model, DataLoader(self.test_data, batch_size=batch_size, shuffle=False, pin_memory=True), nn.CrossEntropyLoss(), device)

        self.round_losses.append(loss)
        self.round_accuracies.append(accuracy)
        print(f"Round {round}/{num_rounds} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        data_to_save = {
          'round_losses': self.round_losses,
          'round_accuracies': self.round_accuracies,
          'selected_clients_per_round': [[client for client in round_clients] for round_clients in self.selected_clients_per_round],  # Serializziamo solo i client_id
        }

        save_checkpoint(self.model, None, round , hyperparameters, "Federated_Uniform/", data_to_save)



    print("Evaluation on test set...")
    loss, accuracy = evaluate_model(self.model, DataLoader(self.test_data, batch_size=batch_size, shuffle=False, pin_memory=True), nn.CrossEntropyLoss(), device)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

     # Plot dei risultati
    plt.figure(figsize=(16, 10))

        # Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(0, num_rounds, 10), self.round_losses, label='Validation Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Round')
    plt.legend()

        # Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(0, num_rounds, 10), self.round_accuracies, label='Validation Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Round')
    plt.legend()


    plt.tight_layout()
    file_name = f"CIFAR100_fedavg_uniform_{hyperparameters}.jpg"
    plt.savefig(file_name)
    plt.show()

    plot_selected_clients_distribution(self.selected_clients_per_round, len(self.clients), hyperparameters)

