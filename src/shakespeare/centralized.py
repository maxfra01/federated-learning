import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import save_checkpoint, load_checkpoint

def train_model(model, train_loader, validation_loader, test_loader, optimizer, scheduler, criterion, epochs, hyperparameters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses, validation_losses, validation_accuracies= [], [], []

    # Load checkpoint if exists
    start_epoch, json_data = load_checkpoint(model, optimizer, hyperparameters)
    if json_data is not None:
        validation_losses = json_data.get('validation_losses', [])
        validation_accuracies = json_data.get('validation_accuracies', [])
        train_losses = json_data.get('train_losses', [])

    if start_epoch >= epochs:
        print(f"Checkpoint found, configuration already completed. Evaluating only on validation set.")
        validation_loss, validation_accuracy = evaluate_model(model, validation_loader, criterion, device)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        return train_losses, validation_losses, validation_accuracies

    # Main training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        total_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            try:
                # Ensure inputs and targets are within valid range
                if inputs.max() >= model.vocab_size or inputs.min() < 0:
                    print(f"Invalid input indices found. Max: {inputs.max()}, Min: {inputs.min()}")
                    continue

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs, _ = model(inputs)  # Ignora gli hidden states


                # Reshape for loss calculation
                batch_size, seq_len, vocab_size = outputs.shape
                outputs = outputs.view(-1, vocab_size)
                targets = targets.view(-1)

                # Create mask for non-padding tokens
                non_pad_mask = targets != 0

                # Only compute loss on non-padding tokens
                valid_outputs = outputs[non_pad_mask]
                valid_targets = targets[non_pad_mask]

                if len(valid_targets) > 0:  # Only compute loss if we have valid tokens
                    loss = criterion(valid_outputs, valid_targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    total_batches += 1

            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        if total_batches > 0:
            epoch_loss = epoch_loss / total_batches

        scheduler.step()

        # Validation
        validation_loss, validation_accuracy = evaluate_model(model, validation_loader, criterion, device)
        train_losses.append(epoch_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)


        print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, hyperparameters, "Centralized/",
            data_to_save={
                'validation_losses': validation_losses,
                'validation_accuracies': validation_accuracies,
                'train_losses': train_losses
            }
        )

    # Final evaluation on test set
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, validation_losses, validation_accuracies

def evaluate_model(model, test_loader, criterion, device):
    total_loss = 0
    correct = 0
    total = 0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            try:
                if inputs.max() >= model.vocab_size or inputs.min() < 0:
                    continue

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs,_ = model(inputs)
                batch_size, seq_len, vocab_size = outputs.shape
                outputs = outputs.view(-1, vocab_size)
                targets = targets.view(-1)

                # Create mask for non-padding tokens
                non_pad_mask = targets != 0
                valid_outputs = outputs[non_pad_mask]
                valid_targets = targets[non_pad_mask]

                if len(valid_targets) > 0:
                    loss = criterion(valid_outputs, valid_targets)
                    total_loss += loss.item()
                    total_batches += 1

                    _, predicted = valid_outputs.max(1)
                    total += valid_targets.size(0)
                    correct += (predicted == valid_targets).sum().item()

            except RuntimeError as e:
                print(f"Error during evaluation: {str(e)}")
                continue

    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy