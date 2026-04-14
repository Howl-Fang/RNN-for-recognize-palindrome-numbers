"""
Student Name: Placeholder Name
Student ID: 0000000
Student Email: student@example.com

Training script for log-uniform 7-digit RNN models with cross-validation
against 7-digit uniform test set from master branch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from generate_data import generate_palindrome_dataset_log_uniform, generate_palindrome_dataset
from model import PalindromeRNN

class PalindromeDataset(Dataset):
    """Dataset for palindrome classification."""
    
    def __init__(self, numbers, labels, max_length=7):
        self.numbers = numbers
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.numbers)
    
    def __getitem__(self, idx):
        num = self.numbers[idx]
        label = self.labels[idx]
        
        # Convert number to sequence of digits
        digits = [int(d) for d in str(num)]
        
        # Pad to max_length
        digits = [0] * (self.max_length - len(digits)) + digits
        
        return torch.tensor(digits, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

class Trainer:
    """Training and evaluation manager."""
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_digits, batch_labels in train_loader:
            batch_digits = batch_digits.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_digits).squeeze()
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_digits, batch_labels in val_loader:
                batch_digits = batch_digits.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_digits).squeeze()
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=30, patience=5):
        """Complete training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_history.append({'loss': train_loss, 'accuracy': train_acc})
            self.val_history.append({'loss': val_loss, 'accuracy': val_acc})
            
            if (epoch + 1) % 5 == 0 or epoch < 3:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_digits, batch_labels in test_loader:
                batch_digits = batch_digits.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_digits).squeeze()
                predictions = (outputs > 0.5).float()
                
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        accuracy = correct / total
        return accuracy

def plot_training_history(train_history, val_history, size, dist_type="loguniform"):
    """Plot training history."""
    os.makedirs('plots', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_history) + 1)
    train_losses = [h['loss'] for h in train_history]
    train_accs = [h['accuracy'] for h in train_history]
    val_losses = [h['loss'] for h in val_history]
    val_accs = [h['accuracy'] for h in val_history]
    
    ax1.plot(epochs, train_losses, label='Training Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss - 7-digit {dist_type} {size}')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, label='Training Accuracy', marker='o')
    ax2.plot(epochs, val_accs, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Training Accuracy - 7-digit {dist_type} {size}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/loss_accuracy_7digit_{dist_type}_{size}.png', dpi=100, bbox_inches='tight')
    plt.close()

def train_log_uniform_models():
    """Train RNN on log-uniform 7-digit distribution."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    sizes = [200, 1000, 50000]
    models_info = {}
    
    for size in sizes:
        print(f"{'='*60}")
        print(f"Training 7-digit log-uniform RNN with {size} samples")
        print(f"{'='*60}")
        
        # Generate log-uniform 7-digit training data
        X, y = generate_palindrome_dataset_log_uniform(size, min_digits=1, max_digits=7)
        
        print(f"Generated {len(X)} log-uniform samples (1-7 digits)")
        n_palindromes = np.sum(y)
        print(f"  Palindromes: {n_palindromes}, Non-palindromes: {len(y) - n_palindromes}\n")
        
        # Create dataset and dataloaders
        dataset = PalindromeDataset(X, y, max_length=7)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create and train model
        model = PalindromeRNN(vocab_size=10, embedding_dim=16, hidden_dim=32, n_layers=2, dropout=0.3)
        model = model.to(device)
        
        trainer = Trainer(model, device, learning_rate=0.001)
        trainer.train(train_loader, val_loader, epochs=30, patience=5)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/model_7digit_loguniform_{size}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"✓ Saved model to {model_path}")
        
        # Plot training history
        plot_training_history(trainer.train_history, trainer.val_history, size, "loguniform")
        
        # Save training log
        os.makedirs('logs', exist_ok=True)
        log_data = {
            'size': size,
            'distribution': 'log_uniform_7digit',
            'train_history': trainer.train_history,
            'val_history': trainer.val_history,
            'final_train_loss': trainer.train_history[-1]['loss'],
            'final_train_acc': trainer.train_history[-1]['accuracy'],
            'final_val_loss': trainer.val_history[-1]['loss'],
            'final_val_acc': trainer.val_history[-1]['accuracy'],
        }
        with open(f'logs/training_log_7digit_loguniform_{size}.json', 'w') as f:
            json.dump(log_data, f, indent=2)
        
        models_info[size] = {
            'model_path': model_path,
            'model': model,
            'final_val_acc': trainer.val_history[-1]['accuracy'],
            'final_train_acc': trainer.train_history[-1]['accuracy'],
        }
        print()
    
    return models_info

def cross_validate(models_info):
    """
    Cross-validate models:
    - Log-uniform models on uniform test set
    - Uniform models on log-uniform test set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print("Cross-Validation: Different Distributions")
    print(f"{'='*60}\n")
    
    # Generate uniform 7-digit test set
    X_uniform, y_uniform = generate_palindrome_dataset(1000, max_digits=7)
    os.makedirs('test_sets', exist_ok=True)
    df_uniform = pd.DataFrame({'number': X_uniform, 'label': y_uniform})
    df_uniform.to_csv('test_sets/test_7digit_uniform.csv', index=False, header=False)
    
    # Generate log-uniform 7-digit test set
    X_loguniform, y_loguniform = generate_palindrome_dataset_log_uniform(1000, min_digits=1, max_digits=7)
    df_loguniform = pd.DataFrame({'number': X_loguniform, 'label': y_loguniform})
    df_loguniform.to_csv('test_sets/test_7digit_loguniform.csv', index=False, header=False)
    
    test_uniform_dataset = PalindromeDataset(X_uniform, y_uniform, max_length=7)
    test_uniform_loader = DataLoader(test_uniform_dataset, batch_size=32, shuffle=False)
    
    test_loguniform_dataset = PalindromeDataset(X_loguniform, y_loguniform, max_length=7)
    test_loguniform_loader = DataLoader(test_loguniform_dataset, batch_size=32, shuffle=False)
    
    results = {}
    
    # Test log-uniform trained models on uniform test set
    print("LOG-UNIFORM TRAINED MODELS on UNIFORM TEST:")
    for size in [200, 1000, 50000]:
        model = models_info[size]['model']
        trainer = Trainer(model, device)
        accuracy = trainer.evaluate(test_uniform_loader)
        
        print(f"  Model {size}: {accuracy:.4f} accuracy")
        
        key = f"loguniform_{size}_on_uniform"
        results[key] = {
            'trained_on': 'log-uniform 7-digit',
            'tested_on': 'uniform 7-digit',
            'accuracy': float(accuracy),
            'train_acc': models_info[size]['final_val_acc'],
            'difference': models_info[size]['final_val_acc'] - accuracy,
        }
    
    # Load uniform trained models from master and test on log-uniform
    print("\nUNIFORM TRAINED MODELS on LOG-UNIFORM TEST:")
    
    for size in [200, 1000, 50000]:
        try:
            model = PalindromeRNN(vocab_size=10, embedding_dim=16, hidden_dim=32, n_layers=2, dropout=0.3)
            model_path = f'models/model_{size}.pt'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                
                trainer = Trainer(model, device)
                accuracy = trainer.evaluate(test_loguniform_loader)
                
                # Get training accuracy from logs
                with open(f'logs/training_log_{size}.json', 'r') as f:
                    log_data = json.load(f)
                    train_acc = log_data.get('final_val_acc', 0)
                
                print(f"  Model {size}: {accuracy:.4f} accuracy")
                
                key = f"uniform_{size}_on_loguniform"
                results[key] = {
                    'trained_on': 'uniform 7-digit',
                    'tested_on': 'log-uniform 7-digit',
                    'accuracy': float(accuracy),
                    'train_acc': train_acc,
                    'difference': train_acc - accuracy,
                }
        except Exception as e:
            print(f"  Model {size}: Not available ({e})")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/cross_validation_7digit_distributions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Cross-Validation Results Summary")
    print(f"{'='*60}")
    
    for key, data in results.items():
        print(f"\n{key}:")
        print(f"  Trained: {data['trained_on']}")
        print(f"  Tested: {data['tested_on']}")
        print(f"  Training Accuracy: {data['train_acc']:.4f}")
        print(f"  Test Accuracy: {data['accuracy']:.4f}")
        print(f"  Difference: {data['difference']:+.4f}")
    
    return results

if __name__ == '__main__':
    # Train on log-uniform 7-digit
    print("PHASE 1: TRAINING LOG-UNIFORM MODELS\n")
    models_info = train_log_uniform_models()
    
    # Cross-validate
    print("\nPHASE 2: CROSS-VALIDATION")
    cross_val_results = cross_validate(models_info)
    
    print("\n" + "="*60)
    print("✓ Training and cross-validation complete!")
    print("="*60)
