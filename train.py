"""
Student Name: Placeholder Name
Student ID: 0000000
Student Email: student@example.com

Training script for palindrome RNN models.
Trains 3 models with 200, 1000, and 50000 examples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple
import os

from generate_data import generate_palindrome_dataset, save_dataset
from model import PalindromeRNN, create_data_loaders, get_device

class Trainer:
    """Trainer class for RNN models."""
    
    def __init__(self, model: nn.Module, device: torch.device, learning_rate: float = 0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x).squeeze()
            
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            pred_labels = (predictions > 0.5).long()
            correct += (pred_labels == batch_y.long()).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x).squeeze()
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                
                pred_labels = (predictions > 0.5).long()
                correct += (pred_labels == batch_y.long()).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 20, patience: int = 5) -> Dict:
        """Train the model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def plot_metrics(self, save_path: str = None):
        """Plot training metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.train_accuracies, label='Train Accuracy')
        axes[1].plot(self.val_accuracies, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

def train_model(dataset_size: int, epochs: int = 20) -> Dict:
    """Train a model on a specific dataset size."""
    
    print(f"\n{'='*60}")
    print(f"Training model on {dataset_size} examples")
    print(f"{'='*60}")
    
    # Generate data
    X, y = generate_palindrome_dataset(dataset_size)
    
    # Split into train and validation (80-20 split)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Initialize model
    device = get_device()
    model = PalindromeRNN()
    
    # Train
    trainer = Trainer(model, device, learning_rate=0.001)
    metrics = trainer.train(train_loader, val_loader, epochs=epochs, patience=5)
    
    # Save model
    model_path = f'models/model_{dataset_size}.pt'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Plot and save
    plot_path = f'plots/loss_accuracy_{dataset_size}.png'
    os.makedirs('plots', exist_ok=True)
    trainer.plot_metrics(plot_path)
    
    # Save training log
    log_path = f'logs/training_log_{dataset_size}.json'
    os.makedirs('logs', exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump({
            'dataset_size': dataset_size,
            'final_train_loss': float(trainer.train_losses[-1]),
            'final_val_loss': float(trainer.val_losses[-1]),
            'final_train_accuracy': float(trainer.train_accuracies[-1]),
            'final_val_accuracy': float(trainer.val_accuracies[-1]),
            'epochs_trained': len(trainer.train_losses),
            # 'hyperparameters': {
            #     'embedding_dim': 16,
            #     'hidden_dim': 32,
            #     'n_layers': 2,
            #     'dropout': 0.3,
            #     'learning_rate': 0.001,
            #     'batch_size': 32
            # }
            'hyperparameters': model.hyperparameters
        }, f, indent=2)
    print(f"Saved training log to {log_path}")
    
    return {
        'dataset_size': dataset_size,
        'model_path': model_path,
        'model': model,
        'device': device,
        'trainer': trainer
    }

def main():
    """Train all three models."""
    dataset_sizes = [200, 1000, 50000]
    results = []
    
    for size in dataset_sizes:
        result = train_model(size, epochs=30)
        results.append(result)

    hyperparameters_path = 'logs/hyperparameters.json'
    with open(hyperparameters_path, 'w') as f:
        json.dump(
            results[0]['model'].hyperparameters,
        f, indent=2)
    print(f"Saved hyperparameters to {hyperparameters_path}")
    
    print(f"\n{'='*60}")
    print("All models trained successfully!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()