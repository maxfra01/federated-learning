import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

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
    trainloader = DataLoader(self.data, batch_size=batch_size, shuffle=True,  pin_memory=True)

    steps = 0
    while steps < local_steps:
      for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = self.model(inputs)  # Ignora gli hidden states

        batch_size, seq_len, vocab_size = outputs.shape
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)

        non_pad_mask = targets != 0

        # Only compute loss on non-padding tokens
        valid_outputs = outputs[non_pad_mask]
        valid_targets = targets[non_pad_mask]

        if len(valid_targets) > 0:  # Only compute loss if we have valid tokens
          loss = criterion(valid_outputs, valid_targets)
          loss.backward()
          optimizer.step()
          steps += 1
        if steps >= local_steps:
          break

    return self.model.state_dict()
