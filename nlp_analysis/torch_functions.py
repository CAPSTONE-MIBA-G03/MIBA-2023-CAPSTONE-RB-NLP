"""
Contains functions for training and testing a PyTorch model.
Inspired and adapted from mrdbourke's Deep Learning with TensorFlow 2.0 course
"""
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

# from tqdm import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Trains an autoencoder for a single epoch.

    Args:
    ----
    model: The autoencoder model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize (e.g., nn.MSELoss).
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g., "cuda" or "cpu").

    Returns:
    ----
    The average training loss (MSE) for the epoch.
    """
    # Put model in train mode
    model.train()

    # Setup train loss value
    train_loss = 0

    # Loop through data loader data batches
    for batch, inputs in enumerate(dataloader):
        # Send data to target device
        inputs = inputs.to(device)

        # 1. Forward pass
        outputs = model(inputs)

        # 2. Calculate and accumulate loss
        loss = loss_fn(outputs, inputs)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust loss to get average loss per batch
    train_loss = train_loss / len(dataloader)

    return train_loss


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Trains and tests an autoencoder model.
    """

    results = {"train_loss": []}

    # Loop through training and testing steps for a number of epochs

    for epoch in range(epochs):
        loss = train_step(
            model=model.to(device), dataloader=dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device
        )

        # Print out what's happening
        print(f"Epoch: {epoch + 1} | loss: {loss:.4f}")

        results["train_loss"].append(loss)

    return results


import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_shape: int, hidden_units: list, output_shape: int, dropout_rate: float = 0.0) -> None:
        """
        Autoencoder model with specified input shape, hidden units, and output shape.

        Args:
        ----
            input_shape (int):
                The size of the input features.
            hidden_units (list):
                A list of integers representing the number of units in each of the hidden layers.
            output_shape (int):
                The size of the output features.
            dropout_rate (float):
                Dropout rate to apply between hidden layers. Default is 0.0 (no dropout).
        """
        super(Autoencoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_shape, hidden_units[0]),
            nn.Tanh(),
            nn.Linear(hidden_units[0], hidden_units[1]),
        )

        self.decode = nn.Sequential(
            nn.Linear(hidden_units[1], hidden_units[0]),
            nn.Tanh(),
            nn.Linear(hidden_units[0], output_shape),
        )

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
