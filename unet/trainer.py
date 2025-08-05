import argparse
import torch
from torch.utils.data import DataLoader
from unet.dataset import UNetDataset
from unet.model import UNet
from tqdm import trange
import numpy as np

import torch.nn as nn
import torch.optim as optim


def train(args):
    # Dataset and DataLoader
    train_dataset = UNetDataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"Using device: {device}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of batches per epoch: {len(train_loader)}")

    # Training loop
    for epoch in trange(args.max_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets, valid_mask) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs + inputs
            # loss = criterion(outputs, targets)
            # Apply valid mask to outputs and targets
            loss = criterion(outputs[valid_mask], targets[valid_mask])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{args.max_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.max_epochs}] Average Loss: {avg_loss:.4f}")

        # Save model weights after each epoch
        torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    train(args)