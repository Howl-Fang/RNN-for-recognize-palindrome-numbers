"""
Evaluation script for testing trained models on test.csv
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import json
import os

from model import PalindromeRNN, PalindromeDataset, get_device
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a trained model."""
    model = PalindromeRNN(vocab_size=10, embedding_dim=16, hidden_dim=32, n_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_on_test_file(model: nn.Module, test_file: str, device: torch.device) -> dict:
    """Evaluate model on test.csv file."""
    
    # Load test data
    df = pd.read_csv(test_file, header=None, names=['number', 'label'])
    X_test = df['number'].values
    y_test = df['label'].values
    
    # Create dataset
    test_dataset = PalindromeDataset(X_test, y_test)
    
    # Get predictions
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for x, y in test_dataset:
            x = x.unsqueeze(0).to(device)
            pred = model(x).squeeze().item()
            predictions.append(1 if pred > 0.5 else 0)
            true_labels.append(y.item())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'total_samples': int(len(predictions))
    }
    
    return results

def main():
    """Evaluate all three models on test.csv"""
    
    if not Path('test.csv').exists():
        print("test.csv not found. Creating sample test file...")
        # Create sample test data
        sample_tests = [
            (123, 0),
            (121, 1),
            (13543, 0),
            (13531, 1),
        ]
        df = pd.DataFrame(sample_tests, columns=['number', 'label'])
        df.to_csv('test.csv', index=False, header=False)
        print("Created test.csv with sample data")
    
    device = get_device()
    dataset_sizes = [200, 1000, 50000]
    all_results = {}
    
    print(f"\n{'='*60}")
    print("Evaluating all models on test.csv")
    print(f"{'='*60}\n")
    
    for size in dataset_sizes:
        model_path = f'models/model_{size}.pt'
        
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}, skipping...")
            continue
        
        print(f"Evaluating model trained on {size} examples...")
        model = load_model(model_path, device)
        results = evaluate_on_test_file(model, 'test.csv', device)
        
        all_results[f'model_{size}'] = results
        
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  Confusion Matrix: TP={results['confusion_matrix']['true_positives']}, "
              f"TN={results['confusion_matrix']['true_negatives']}, "
              f"FP={results['confusion_matrix']['false_positives']}, "
              f"FN={results['confusion_matrix']['false_negatives']}")
        print()
    
    # Save evaluation results
    # os.makedirs('results', exist_ok=True)
    # with open('results/evaluation_results.json', 'w') as f:
    #     json.dump(all_results, f, indent=2)
    # print(f"Saved evaluation results to results/evaluation_results.json")
    
    return all_results

if __name__ == '__main__':
    main()