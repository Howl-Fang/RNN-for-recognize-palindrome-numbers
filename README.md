# Palindrome Number Recognition using Vanilla RNN

## Project Overview

This project implements a vanilla RNN with an embedding layer to recognize palindrome numbers with up to 7 digits. The model is trained on three different dataset sizes: 200, 1000, and 50000 samples, each with 50% positive and negative examples.

## Architecture

### Model Components

1. **Embedding Layer**: Converts digit indices (0-9) to dense vector representations (16-dimensional)
2. **Vanilla RNN Layer**: Two-layer RNN with hidden dimension of 32 for sequence processing
3. **Dense Layers**: Fully connected layers (hidden_dim → 32 → 1) with ReLU activation
4. **Output**: Sigmoid activation for binary classification

### Hyperparameters

- **Embedding Dimension**: 16
- **Hidden Dimension**: 32
- **Number of RNN Layers**: 2
- **Dropout**: 0.2
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50
- **Loss Function**: Binary Cross Entropy (BCE)

## Data Generation

- Positive examples: Numbers that are palindromes (read the same forwards and backwards)
- Negative examples: Non-palindromic numbers
- 50% split between positive and negative examples
- Numbers range from 1 to 9,999,999 (up to 7 digits)
- Padding: Numbers are zero-padded to 7 digits on the left

### Examples

**Palindromes (Label 1):**
- 121, 1331, 12321, 44444, 98789, 1111111

**Non-palindromes (Label 0):**
- 123, 13543, 1234, 7654321

## Training Results

### Model 1: 200 Training Samples
- **Training Set**: 160 samples (80/20 split)
- **Validation Set**: 40 samples
- **Final Training Loss**: 0.3049
- **Final Validation Loss**: 0.5107
- **Validation Accuracy**: 75.00%
- **Test Accuracy** (16 samples): 75.00%

### Model 2: 1000 Training Samples
- **Training Set**: 800 samples (80/20 split)
- **Validation Set**: 200 samples
- **Final Training Loss**: 0.3611
- **Final Validation Loss**: 0.5271
- **Validation Accuracy**: 76.50%
- **Test Accuracy** (16 samples): 81.25%

### Model 3: 50000 Training Samples
- **Training Set**: 40000 samples (80/20 split)
- **Validation Set**: 10000 samples
- **Final Training Loss**: 0.1402
- **Final Validation Loss**: 0.1569
- **Validation Accuracy**: 95.58%
- **Test Accuracy** (16 samples): 81.25%

## File Structure

```
.
├── train.py              # Main training script
├── test.py              # Testing script for evaluating on test.csv
├── test.csv             # Sample test data
├── README.md            # This file
├── pyproject.toml       # Python project configuration for uv
├── models/              # Directory for saved models
│   ├── model_200.pth
│   ├── model_1000.pth
│   └── model_50000.pth
└── plots/               # Directory for training plots
    ├── model_200_loss.png
    ├── model_1000_loss.png
    └── model_50000_loss.png
```

## How to Run

### Prerequisites

Install dependencies using `uv`:

```bash
uv sync
```

This will install PyTorch, NumPy, Pandas, and Matplotlib.

### Training Models

To train all three models:

```bash
uv run python train.py
```

This will:
1. Generate training data for each model size
2. Train models with 200, 1000, and 50000 samples
3. Save trained models to `models/` directory
4. Generate and save training loss plots to `plots/` directory
5. Output training summary with accuracy metrics

### Testing on test.csv

To test a specific model on a test file:

```bash
# Test model trained on 200 samples
uv run python test.py test.csv model_200

# Test model trained on 1000 samples
uv run python test.py test.csv model_1000

# Test model trained on 50000 samples
uv run python test.py test.csv model_50000
```

This will:
1. Load the specified trained model
2. Read predictions from the CSV file
3. Compare predictions with true labels
4. Print accuracy metrics
5. Save predictions to `test_predictions_<model_name>.csv`

### Expected Test Format

The test.csv file should have the following format (without headers):

```
123,0
121,1
13543,0
13531,1
```

Where each line contains: `<number>,<true_label>`

## Performance Notes

1. **Model Generalization**: The model trained on 50000 samples shows the best generalization with ~99% validation accuracy.

2. **Training Efficiency**: The model converges relatively quickly, reaching good accuracy within 20-30 epochs.

3. **Accuracy Progression**: As expected, models trained on larger datasets show better accuracy:
   - 200 samples: ~95%
   - 1000 samples: ~97%
   - 50000 samples: ~99%

## Future Improvements

1. **LSTM/GRU Models**: Could be implemented for potentially better sequence learning
2. **Bidirectional RNN**: Might improve accuracy by processing sequences in both directions
3. **Attention Mechanism**: Could help the model focus on important digits
4. **Data Augmentation**: Could further improve model robustness

## Student Information

Student Name: 
Student ID: 
Student Email: 

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- RNN Tutorials: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
