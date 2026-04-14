"""
Student Name: Placeholder Name
Student ID: 0000000
Student Email: student@example.com

Comprehensive evaluation script for all RNN and LSTM models.
Tests on both 7-digit and 12-digit log-scale data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import json
import os

from model import PalindromeRNN, PalindromeLSTM, PalindromeDataset, get_device
from generate_data import generate_palindrome_dataset, generate_palindrome_dataset_log_uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_rnn_model(model_path: str, device: torch.device, max_length: int = 7) -> nn.Module:
    """Load trained RNN model."""
    model = PalindromeRNN(vocab_size=10, embedding_dim=16, hidden_dim=32, n_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_lstm_model(model_path: str, device: torch.device, max_length: int = 7) -> nn.Module:
    """Load trained LSTM model."""
    model = PalindromeLSTM(vocab_size=10, embedding_dim=16, hidden_dim=32, n_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                   device: torch.device, max_length: int = 7) -> dict:
    """Evaluate model on given dataset."""
    
    test_dataset = PalindromeDataset(X_test, y_test, max_length=max_length)
    
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
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    return {
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

def main():
    """Comprehensive evaluation of all models."""
    
    device = get_device()
    dataset_sizes = [200, 1000, 50000]
    all_results = {}
    
    # Generate test sets
    print("Generating test sets...")
    X_test_7digit, y_test_7digit = generate_palindrome_dataset(1000, max_digits=7, random_seed=42)
    X_test_12digit, y_test_12digit = generate_palindrome_dataset_log_uniform(1000, max_digits=12, random_seed=99)
    
    print(f"\n{'='*70}")
    print("EVALUATION ON 7-DIGIT TEST DATA")
    print(f"{'='*70}")
    
    print("\nVanilla RNN Models (Original Training):")
    for size in dataset_sizes:
        model_path = f'models/model_{size}.pt'
        if Path(model_path).exists():
            print(f"\n  Model RNN-{size}:")
            model = load_rnn_model(model_path, device, max_length=7)
            results = evaluate_model(model, X_test_7digit, y_test_7digit, device, max_length=7)
            all_results[f'rnn_7digit_{size}'] = results
            
            print(f"    Accuracy:  {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall:    {results['recall']:.4f}")
            print(f"    F1-Score:  {results['f1_score']:.4f}")
    
    print("\n\nLSTM Models (7-Digit Training):")
    for size in dataset_sizes:
        model_path = f'models/lstm/lstm_7digit_{size}.pt'
        if Path(model_path).exists():
            print(f"\n  Model LSTM-{size}:")
            model = load_lstm_model(model_path, device, max_length=7)
            results = evaluate_model(model, X_test_7digit, y_test_7digit, device, max_length=7)
            all_results[f'lstm_7digit_{size}'] = results
            
            print(f"    Accuracy:  {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall:    {results['recall']:.4f}")
            print(f"    F1-Score:  {results['f1_score']:.4f}")
    
    print(f"\n{'='*70}")
    print("EVALUATION ON 12-DIGIT LOG-SCALE TEST DATA")
    print(f"{'='*70}")
    
    print("\nVanilla RNN Models (7-Digit Training, 12-Digit Test):")
    for size in dataset_sizes:
        model_path = f'models/model_{size}.pt'
        if Path(model_path).exists():
            print(f"\n  Model RNN-{size}:")
            model = load_rnn_model(model_path, device, max_length=12)
            results = evaluate_model(model, X_test_12digit, y_test_12digit, device, max_length=12)
            all_results[f'rnn_12digit_crosseval_{size}'] = results
            
            print(f"    Accuracy:  {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall:    {results['recall']:.4f}")
            print(f"    F1-Score:  {results['f1_score']:.4f}")
    
    print("\n\nLSTM Models (7-Digit Training, 12-Digit Test):")
    for size in dataset_sizes:
        model_path = f'models/lstm/lstm_7digit_{size}.pt'
        if Path(model_path).exists():
            print(f"\n  Model LSTM-{size}:")
            model = load_lstm_model(model_path, device, max_length=12)
            results = evaluate_model(model, X_test_12digit, y_test_12digit, device, max_length=12)
            all_results[f'lstm_7digit_12digit_crosseval_{size}'] = results
            
            print(f"    Accuracy:  {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall:    {results['recall']:.4f}")
            print(f"    F1-Score:  {results['f1_score']:.4f}")
    
    print("\n\nLSTM Models (12-Digit Training, 12-Digit Test):")
    for size in dataset_sizes:
        model_path = f'models/lstm/lstm_12digit_logscale_{size}.pt'
        if Path(model_path).exists():
            print(f"\n  Model LSTM-12digit-{size}:")
            model = load_lstm_model(model_path, device, max_length=12)
            results = evaluate_model(model, X_test_12digit, y_test_12digit, device, max_length=12)
            all_results[f'lstm_12digit_{size}'] = results
            
            print(f"    Accuracy:  {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall:    {results['recall']:.4f}")
            print(f"    F1-Score:  {results['f1_score']:.4f}")
    
    # Save comprehensive results
    os.makedirs('results', exist_ok=True)
    result_path = 'results/comprehensive_evaluation.json'
    with open(result_path, 'w') as f:
        json.dump({
            'test_sets': {
                '7digit': {
                    'samples': len(X_test_7digit),
                    'generation': 'original',
                    'max_digits': 7
                },
                '12digit_logscale': {
                    'samples': len(X_test_12digit),
                    'generation': 'log_uniform',
                    'max_digits': 12
                }
            },
            'model_results': all_results
        }, f, indent=2)
    print(f"\n\nSaved comprehensive evaluation results to {result_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}\n")
    
    print("7-DIGIT TEST DATA:")
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 73)
    for key in sorted(all_results.keys()):
        if '12digit' not in key or 'crosseval' not in key:
            if '7digit' in key:
                results = all_results[key]
                model_name = key.replace('_', '-')
                print(f"{model_name:<25} {results['accuracy']:<12.4f} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['f1_score']:<12.4f}")
    
    print(f"\n12-DIGIT LOG-SCALE TEST DATA:")
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 73)
    for key in sorted(all_results.keys()):
        if 'crosseval' in key or ('lstm_12digit_' in key and 'crosseval' not in key):
            results = all_results[key]
            model_name = key.replace('_', '-')
            print(f"{model_name:<25} {results['accuracy']:<12.4f} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['f1_score']:<12.4f}")

if __name__ == '__main__':
    main()
