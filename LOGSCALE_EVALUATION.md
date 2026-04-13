# 12-Digit Palindrome Classification with Logarithmic Distribution

## Overview
Models trained on 7-digit palindromes are evaluated on a new 12-digit dataset with uniform logarithmic distribution. This tests generalization to larger numbers and uniform digit counts.

## Distribution Design
- **Digit Range**: 1 to 12 digits
- **Distribution**: Uniform in logarithmic space (linear uniform digit count)
- **Purpose**: Each digit count appears equally likely in the dataset

## Test Set Statistics
- **Total Samples**: 1000
- **Class Split**: 500 palindromes (50%), 500 non-palindromes (50%)
- **Digit Distribution**:
  - 1 digit: 49 samples (4.9%)
  - 2 digits: 77 samples (7.7%)
  - 3 digits: 85 samples (8.5%)
  - 4 digits: 90 samples (9.0%)
  - 5 digits: 85 samples (8.5%)
  - 6 digits: 80 samples (8.0%)
  - 7 digits: 95 samples (9.5%)
  - 8 digits: 92 samples (9.2%)
  - 9 digits: 82 samples (8.2%)
  - 10 digits: 91 samples (9.1%)
  - 11 digits: 75 samples (7.5%)
  - 12 digits: 99 samples (9.9%)
- **Mean Digit Count**: 6.79 (±3.34)

## Model Performance on 12-Digit Log-Scale Data

### Model 200 (trained on 200 7-digit examples)
- **Accuracy**: 46.00%
- **Precision**: 46.60%
- **Recall**: 54.80%
- **F1-Score**: 50.37%
- **TP**: 274, **TN**: 186, **FP**: 314, **FN**: 226
- **Status**: Struggles with generalization; many false positives and false negatives

### Model 1000 (trained on 1000 7-digit examples)
- **Accuracy**: 57.00%
- **Precision**: 59.67%
- **Recall**: 43.20%
- **F1-Score**: 50.12%
- **TP**: 216, **TN**: 354, **FP**: 146, **FN**: 284
- **Status**: Better balance; more conservative in palindrome predictions

### Model 50000 (trained on 50000 7-digit examples)
- **Accuracy**: 54.00%
- **Precision**: 82.26%
- **Recall**: 10.20%
- **F1-Score**: 18.15%
- **TP**: 51, **TN**: 489, **FP**: 11, **FN**: 449
- **Status**: Very conservative - rarely predicts palindromes (high precision, very low recall)

## Key Findings

1. **Overfitting to Input Range**: Model 50000 (best on 7-digit data) performs poorly on 12-digit data, achieving only 54% accuracy. This suggests the model overfits to the magnitude range of the training data.

2. **Scale Generalization Issue**: The larger the model (more training data), the worse it generalizes to larger numbers. This is likely because:
   - The model learned patterns specific to 7-digit numbers
   - The embedding layer doesn't generalize well to new digit values
   - The RNN hidden state initialization assumes a certain input magnitude

3. **Model 1000 Best Generalization**: Model 1000 shows the best balanced generalization with 57% accuracy and reasonable precision/recall trade-off.

4. **Palindrome Detection Difficulty**: Even the models struggle to detect palindromes at scale:
   - Model 200 overpredicts palindromes (54.8% recall, but many false positives)
   - Model 50000 underpredicts palindromes (only 10.2% recall, 449 false negatives out of 500)

## Next Steps
- Train new models specifically on 12-digit data with log-scale distribution
- Consider architecture modifications for better scale generalization
- Investigate if positional encoding or normalization helps
