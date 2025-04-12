import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Client:

  def __init__(self, model, client_id, data, optimizer_params):
    self.client_id = client_id
    self.data = data
    self.model = model
    self.optimizer_params = optimizer_params

  def train(self, global_weights, local_steps, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)
    self.model.load_state_dict(global_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        self.model.parameters(),
        lr=self.optimizer_params['lr'],
        momentum=self.optimizer_params['momentum'],
        weight_decay=self.optimizer_params['weight_decay']
        )
    trainloader = DataLoader(self.data, batch_size=batch_size, pin_memory=True, shuffle=True)
    steps = 0  # Track the number of steps
    while steps < local_steps:
      for inputs, targets in trainloader:
          if steps >= local_steps:  # Stop after completing the required steps
              break
          inputs, targets = inputs.to(device), targets.to(device)
          optimizer.zero_grad()
          outputs = self.model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()
          steps += 1
    return self.model.state_dict()