"""
Student Name: Placeholder Name
Student ID: 0000000
Student Email: student@example.com

Dataset generation for palindrome number classification task.
Generates balanced training datasets with specified sizes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

def is_palindrome(n: int) -> bool:
    """Check if a number is a palindrome."""
    s = str(n)
    return s == s[::-1]

def generate_palindrome_dataset(n_samples: int, max_digits: int = 7, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced dataset of palindrome classification.
    
    Args:
        n_samples: Total number of samples (will be split 50/50 pos/neg)
        max_digits: Maximum number of digits (0 to 10^max_digits - 1)
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Array of numbers (shape: [n_samples])
        y: Array of labels (shape: [n_samples]), 1 for palindrome, 0 otherwise
    """
    np.random.seed(random_seed)
    
    max_val = 10 ** max_digits - 1
    n_per_class = n_samples // 2
    
    X_palindrome = []
    X_non_palindrome = []
    
    # Generate palindromes by constructing them directly
    for _ in range(n_per_class * 2):  # Try to generate more than needed
        # Randomly construct a palindrome
        half_len = (max_digits + 1) // 2
        half_digits = [np.random.randint(0, 10) for _ in range(half_len)]
        
        if max_digits % 2 == 0:
            # Even length: mirror the first half
            palindrome_digits = half_digits + half_digits[::-1]
        else:
            # Odd length: mirror the first half except the middle digit
            palindrome_digits = half_digits + half_digits[-2::-1]
        
        num = int(''.join(map(str, palindrome_digits)))
        if num <= max_val and num not in X_palindrome:
            X_palindrome.append(num)
            if len(X_palindrome) >= n_per_class:
                break
    
    # Generate non-palindromes
    attempts = 0
    max_attempts = n_per_class * 100
    while len(X_non_palindrome) < n_per_class and attempts < max_attempts:
        num = np.random.randint(0, max_val + 1)
        if not is_palindrome(num):
            X_non_palindrome.append(num)
        attempts += 1
    
    # Ensure we have exactly n_per_class samples
    X_palindrome = X_palindrome[:n_per_class]
    X_non_palindrome = X_non_palindrome[:n_per_class]
    
    # If we don't have enough non-palindromes, generate more
    if len(X_non_palindrome) < n_per_class:
        for num in range(max_val + 1):
            if len(X_non_palindrome) >= n_per_class:
                break
            if not is_palindrome(num):
                X_non_palindrome.append(num)
    
    X = np.array(X_palindrome + X_non_palindrome)
    y = np.array([1] * len(X_palindrome) + [0] * len(X_non_palindrome))
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """Save dataset to CSV file."""
    df = pd.DataFrame({'number': X, 'label': y})
    df.to_csv(filepath, index=False, header=False)
    print(f"Saved {len(X)} samples to {filepath}")

def main():
    """Generate all three training datasets."""
    sizes = [200, 1000, 50000]
    
    for size in sizes:
        X, y = generate_palindrome_dataset(size)
        filepath = f'train_{size}.csv'
        save_dataset(X, y, filepath)
        
        # Print statistics
        n_palindromes = np.sum(y)
        n_non_palindromes = len(y) - n_palindromes
        print(f"Dataset size {size}: {n_palindromes} palindromes, {n_non_palindromes} non-palindromes")

if __name__ == '__main__':
    main()
