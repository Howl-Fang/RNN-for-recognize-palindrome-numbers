"""
Student Name: 
Student ID: 
Student Email: 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json


def is_palindrome(n):
    """Check if a number is a palindrome."""
    s = str(n)
    return s == s[::-1]


def generate_data(num_samples, seed=42):
    """Generate palindrome classification data with 50% positive and negative examples."""
    np.random.seed(seed)
    data = []
    labels = []
    
    # Generate equal positive and negative examples
    positive_count = num_samples // 2
    negative_count = num_samples - positive_count
    
    # Generate positive examples (palindromes)
    generated_positive = 0
    while generated_positive < positive_count:
        num = np.random.randint(1, 10000000)  # up to 7 digits
        if is_palindrome(num):
            data.append(num)
            labels.append(1)
            generated_positive += 1
    
    # Generate negative examples (non-palindromes)
    generated_negative = 0
    while generated_negative < negative_count:
        num = np.random.randint(1, 10000000)
        if not is_palindrome(num):
            data.append(num)
            labels.append(0)
            generated_negative += 1
    
    # Shuffle the data
    indices = np.random.permutation(len(data))
    data = [data[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return data, labels


def number_to_sequence(num, max_length=7):
    """Convert a number to a sequence of digits."""
    digits = [int(d) for d in str(num)]
    # Pad with zeros to max_length
    if len(digits) < max_length:
        digits = [0] * (max_length - len(digits)) + digits
    return digits[:max_length]


class PalindromeDataset(Dataset):
    """Dataset for palindrome classification."""
    
    def __init__(self, data, labels, max_length=7):
        self.data = data
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        num = self.data[idx]
        label = self.labels[idx]
        sequence = number_to_sequence(num, self.max_length)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


class PalindromeRNN(nn.Module):
    """Vanilla RNN with embedding layer for palindrome classification."""
    
    def __init__(self, vocab_size=10, embedding_dim=16, hidden_dim=32, num_layers=2, dropout=0.2):
        super(PalindromeRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, 
                         dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        rnn_out, hidden = self.rnn(embedded)  # rnn_out: (batch_size, seq_length, hidden_dim)
        
        # Use the last hidden state
        last_hidden = rnn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        out = self.relu(self.fc(last_hidden))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        
        return out.squeeze()


class PalindromeLSTM(nn.Module):
    """LSTM with embedding layer for palindrome classification."""
    
    def __init__(self, vocab_size=10, embedding_dim=16, hidden_dim=32, num_layers=2, dropout=0.2):
        super(PalindromeLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: (batch_size, seq_length, hidden_dim)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        out = self.relu(self.fc(last_hidden))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        
        return out.squeeze()


def train_model(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.001):
    """Train the model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    model = model.to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy


def main():
    """Main training function."""
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training configurations
    configs = [
        {'num_samples': 200, 'model_name': 'lstm_model_200', 'model_class': PalindromeLSTM},
        {'num_samples': 1000, 'model_name': 'lstm_model_1000', 'model_class': PalindromeLSTM},
        {'num_samples': 50000, 'model_name': 'lstm_model_50000', 'model_class': PalindromeLSTM},
    ]
    
    results = []
    
    for config in configs:
        num_samples = config['num_samples']
        model_name = config['model_name']
        model_class = config['model_class']
        
        print(f"\n{'='*60}")
        print(f"Training LSTM model with {num_samples} samples: {model_name}")
        print(f"{'='*60}")
        
        # Generate data
        print("Generating training data...")
        data, labels = generate_data(num_samples, seed=42)
        
        # Split into train/val (80/20)
        split_idx = int(0.8 * len(data))
        train_data, train_labels = data[:split_idx], labels[:split_idx]
        val_data, val_labels = data[split_idx:], labels[split_idx:]
        
        # Create datasets and dataloaders
        train_dataset = PalindromeDataset(train_data, train_labels)
        val_dataset = PalindromeDataset(val_data, val_labels)
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        model = model_class(vocab_size=10, embedding_dim=16, hidden_dim=32, 
                           num_layers=2, dropout=0.2)
        
        # Train model
        print("Training model...")
        train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                               epochs=50, device=device, learning_rate=0.001)
        
        # Evaluate on validation set
        val_accuracy = evaluate_model(model, val_loader, device=device)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{model_name}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {model_name} ({num_samples} samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = f'plots/{model_name}_loss.png'
        os.makedirs('plots', exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close()
        
        # Store results
        result = {
            'num_samples': num_samples,
            'model_name': model_name,
            'model_type': 'LSTM',
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'val_accuracy': val_accuracy,
            'model_path': model_path,
            'plot_path': plot_path,
        }
        results.append(result)
        
        print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    
    # Save results
    with open('lstm_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("LSTM Training Summary")
    print(f"{'='*60}")
    for result in results:
        print(f"\nModel: {result['model_name']} ({result['num_samples']} samples)")
        print(f"  Final Train Loss: {result['final_train_loss']:.4f}")
        print(f"  Final Val Loss: {result['final_val_loss']:.4f}")
        print(f"  Validation Accuracy: {result['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
