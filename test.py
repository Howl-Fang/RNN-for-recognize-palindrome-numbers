"""
Script to test trained models on test.csv format.
Usage: python test.py <path_to_test_csv> <model_name>
Example: python test.py test.csv model_50000
"""

import torch
import sys
import os
from train import PalindromeRNN, number_to_sequence
import pandas as pd


def load_model(model_name, device='cpu'):
    """Load a trained model."""
    model = PalindromeRNN(vocab_size=10, embedding_dim=16, hidden_dim=32, 
                         num_layers=2, dropout=0.2)
    model_path = f'models/{model_name}.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def test_on_csv(csv_path, model_name):
    """Test model on CSV file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model: {model_name}")
    model = load_model(model_name, device=device)
    
    # Read CSV
    print(f"Reading test file: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=['number', 'label'])
    
    # Test
    correct = 0
    total = len(df)
    predictions = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            num = row['number']
            true_label = row['label']
            
            # Convert to sequence
            sequence = number_to_sequence(num, max_length=7)
            sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            
            # Predict
            output = model(sequence_tensor)
            pred_label = (output > 0.5).float().item()
            
            predictions.append(int(pred_label))
            
            if pred_label == true_label:
                correct += 1
    
    accuracy = correct / total
    
    # Print results
    print(f"\nTest Results for {model_name}:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Save predictions to CSV
    output_path = f'test_predictions_{model_name}.csv'
    df['prediction'] = predictions
    df.to_csv(output_path, index=False, header=False)
    print(f"  Predictions saved to: {output_path}")
    
    return accuracy


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test.py <path_to_test_csv> <model_name>")
        print("Example: python test.py test.csv model_50000")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_name = sys.argv[2]
    
    test_on_csv(csv_path, model_name)
