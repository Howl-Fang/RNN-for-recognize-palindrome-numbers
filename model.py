"""
RNN model with embedding layer for palindrome number classification.
Based on Tutorial 6 architecture.
"""

from gc import enable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
from sys import argv

class PalindromeDataset(Dataset):
    """Custom dataset for palindrome classification."""
    
    def __init__(self, numbers: np.ndarray, labels: np.ndarray, max_length: int = 7):
        """
        Args:
            numbers: Array of numbers
            labels: Array of labels (0 or 1)
            max_length: Maximum length to pad sequences
        """
        self.numbers = numbers
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.numbers)
    
    def __getitem__(self, idx):
        num = self.numbers[idx]
        label = self.labels[idx]
        
        # Convert number to sequence of digits, padded to max_length
        digits = [int(d) for d in str(num)]
        # Pad on the left with 0s
        digits = [0] * (self.max_length - len(digits)) + digits
        
        return torch.tensor(digits, dtype=torch.long), torch.tensor(label, dtype=torch.float)

class PalindromeRNN(nn.Module):
    """Vanilla RNN with embedding layer for palindrome classification."""
    
    def __init__(self, vocab_size: int = 10, embedding_dim: int = 16, hidden_dim: int = 32, 
                 output_dim: int = 1, n_layers: int = 2, dropout: float = 0.3, enableLSTM: bool = True,
                 hyperparameters: dict = None):
        """
        Args:
            vocab_size: Size of vocabulary (digits 0-9)
            embedding_dim: Dimension of embedding layer
            hidden_dim: Dimension of hidden state
            output_dim: Output dimension (1 for binary classification)
            n_layers: Number of RNN layers
            dropout: Dropout rate
            hyperparameters: dictionary to load once at a time, overwrite other hyperparameters if provided
        """
        super().__init__()
        if hyperparameters is not None:
            vocab_size = hyperparameters.get('vocab_size', vocab_size)
            embedding_dim = hyperparameters.get('embedding_dim', embedding_dim)
            hidden_dim = hyperparameters.get('hidden_dim', hidden_dim)
            output_dim = hyperparameters.get('output_dim', output_dim)
            n_layers = hyperparameters.get('n_layers', n_layers)
            dropout = hyperparameters.get('dropout', dropout)
            enableLSTM = hyperparameters.get('enableLSTM', enableLSTM)
        self.hyperparameters = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'n_layers': n_layers,
            'dropout': dropout,
            'enableLSTM': enableLSTM
        }
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if enableLSTM:
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                              dropout=dropout if n_layers > 1 else 0, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, 
                             dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, text):
        """
        Args:
            text: Tensor of shape [batch_size, seq_length]
            
        Returns:
            Predictions of shape [batch_size, 1]
        """
        embedded = self.embedding(text)  # [batch_size, seq_length, embedding_dim]
        if self.hyperparameters['enableLSTM']:
            output, (hidden, cell) = self.rnn(embedded)  # output: [batch_size, seq_length, hidden_dim]
        else:
            output, hidden = self.rnn(embedded)  # output: [batch_size, seq_length, hidden_dim]
        hidden_last = hidden[-1]  # Take last layer's hidden state: [batch_size, hidden_dim]
        logits = self.fc(hidden_last)  # [batch_size, 1]
        predictions = self.sigmoid(logits)
        return predictions

def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray,
                       batch_size: int = 32, max_length: int = 7) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation."""
    
    train_dataset = PalindromeDataset(X_train, y_train, max_length)
    val_dataset = PalindromeDataset(X_val, y_val, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_device():
    """Get device (GPU if available, else CPU)."""
    if argv[-1] == 'dev=cpu':
        print("Running on CPU only (dev=cpu flag detected).")
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_default = PalindromeRNN()
print("Default model architecture:")
print(model_default)