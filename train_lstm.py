"""
Student Name: Placeholder Name
Student ID: 0000000
Student Email: student@example.com

Training script for LSTM models on palindrome classification.
Trains LSTM on both 7-digit and 12-digit log-scale data.
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

from generate_data import generate_palindrome_dataset, generate_palindrome_dataset_log_uniform
from model import PalindromeLSTM, create_data_loaders, get_device

class Trainer:
    """Trainer class for LSTM models."""
    
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
    
    def train(self, train_loader, val_loader, epochs: int = 30, patience: int = 5) -> Dict:
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
        
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
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

def train_lstm_7digit(dataset_size: int, epochs: int = 30) -> Dict:
    """Train LSTM on 7-digit data."""
    
    print(f"\n{'='*60}")
    print(f"Training LSTM on {dataset_size} 7-digit examples")
    print(f"{'='*60}")
    
    X, y = generate_palindrome_dataset(dataset_size, max_digits=7)
    
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32, max_length=7)
    
    device = get_device()
    model = PalindromeLSTM(vocab_size=10, embedding_dim=16, hidden_dim=32, n_layers=2, dropout=0.3)
    
    trainer = Trainer(model, device, learning_rate=0.001)
    metrics = trainer.train(train_loader, val_loader, epochs=epochs, patience=5)
    
    os.makedirs('models/lstm', exist_ok=True)
    model_path = f'models/lstm/lstm_7digit_{dataset_size}.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    os.makedirs('plots/lstm', exist_ok=True)
    plot_path = f'plots/lstm/lstm_7digit_loss_accuracy_{dataset_size}.png'
    trainer.plot_metrics(plot_path)
    
    os.makedirs('logs/lstm', exist_ok=True)
    log_path = f'logs/lstm/lstm_7digit_training_log_{dataset_size}.json'
    with open(log_path, 'w') as f:
        json.dump({
            'data_type': '7-digit',
            'dataset_size': dataset_size,
            'model_type': 'LSTM',
            'final_train_loss': float(trainer.train_losses[-1]),
            'final_val_loss': float(trainer.val_losses[-1]),
            'final_train_accuracy': float(trainer.train_accuracies[-1]),
            'final_val_accuracy': float(trainer.val_accuracies[-1]),
            'epochs_trained': len(trainer.train_losses),
            'hyperparameters': {
                'embedding_dim': 16,
                'hidden_dim': 32,
                'n_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }, f, indent=2)
    print(f"Saved training log to {log_path}")
    
    return {'model': model, 'device': device, 'trainer': trainer, 'model_path': model_path}

def train_lstm_12digit_logscale(dataset_size: int, epochs: int = 30) -> Dict:
    """Train LSTM on 12-digit log-scale data."""
    
    print(f"\n{'='*60}")
    print(f"Training LSTM on {dataset_size} 12-digit log-scale examples")
    print(f"{'='*60}")
    
    X, y = generate_palindrome_dataset_log_uniform(dataset_size, max_digits=12)
    
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32, max_length=12)
    
    device = get_device()
    model = PalindromeLSTM(vocab_size=10, embedding_dim=16, hidden_dim=32, n_layers=2, dropout=0.3)
    
    trainer = Trainer(model, device, learning_rate=0.001)
    metrics = trainer.train(train_loader, val_loader, epochs=epochs, patience=5)
    
    os.makedirs('models/lstm', exist_ok=True)
    model_path = f'models/lstm/lstm_12digit_logscale_{dataset_size}.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    os.makedirs('plots/lstm', exist_ok=True)
    plot_path = f'plots/lstm/lstm_12digit_loss_accuracy_{dataset_size}.png'
    trainer.plot_metrics(plot_path)
    
    os.makedirs('logs/lstm', exist_ok=True)
    log_path = f'logs/lstm/lstm_12digit_training_log_{dataset_size}.json'
    with open(log_path, 'w') as f:
        json.dump({
            'data_type': '12-digit-logscale',
            'dataset_size': dataset_size,
            'model_type': 'LSTM',
            'final_train_loss': float(trainer.train_losses[-1]),
            'final_val_loss': float(trainer.val_losses[-1]),
            'final_train_accuracy': float(trainer.train_accuracies[-1]),
            'final_val_accuracy': float(trainer.val_accuracies[-1]),
            'epochs_trained': len(trainer.train_losses),
            'hyperparameters': {
                'embedding_dim': 16,
                'hidden_dim': 32,
                'n_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }, f, indent=2)
    print(f"Saved training log to {log_path}")
    
    return {'model': model, 'device': device, 'trainer': trainer, 'model_path': model_path}

def main():
    """Train LSTM models on both datasets."""
    dataset_sizes = [200, 1000, 50000]
    
    print("\n" + "="*60)
    print("TRAINING LSTM MODELS ON 7-DIGIT DATA")
    print("="*60)
    
    for size in dataset_sizes:
        train_lstm_7digit(size, epochs=30)
    
    print("\n" + "="*60)
    print("TRAINING LSTM MODELS ON 12-DIGIT LOG-SCALE DATA")
    print("="*60)
    
    for size in dataset_sizes:
        train_lstm_12digit_logscale(size, epochs=30)
    
    print("\n" + "="*60)
    print("All LSTM models trained successfully!")
    print("="*60)

if __name__ == '__main__':
    main()
