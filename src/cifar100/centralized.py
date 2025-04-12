import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils import save_checkpoint, load_checkpoint


def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, epochs, hyperparameters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses, test_losses, test_accuracies = [], [], []

    # Load Checkpoint if exists
    start_epoch, json_data = load_checkpoint(model, optimizer, hyperparameters, "Centralized/")
    if json_data is not None:
        test_losses = json_data.get('test_losses', [])
        test_accuracies = json_data.get('test_accuracies', [])
        train_losses = json_data.get('train_losses', [])

    if start_epoch >= epochs:
        print(f"Checkpoint found, configuration already completed. Evaluation only on the test set.")
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        return train_losses, test_losses, test_accuracies

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, hyperparameters, "Centralized/")

        # Evaluation on the validation set
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        train_losses.append(epoch_loss / len(train_loader))
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_losses, test_accuracies

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return total_loss / len(test_loader), correct / total
