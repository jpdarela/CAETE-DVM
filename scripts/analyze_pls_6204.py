#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze PLS 6204 compared to trait means"""

import pandas as pd
import numpy as np
from pathlib import Path

# Read the CSV file
csv_path = Path("../outputs/pls_attrs-50000.csv")
df = pd.read_csv(csv_path)

# Get the trait columns (excluding PLS_id)
trait_columns = df.columns[1:].tolist()

# Calculate means for all traits
trait_means = df[trait_columns].mean()

# Get PLS 6204 (row index 6204 since row 0 is PLS 0.0)
pls_6204 = df[df['PLS_id'] == 7625].iloc[0]

# Calculate differences and percent differences
print("="*80)
print(f"PLS 6204 TRAIT ANALYSIS")
print("="*80)
print(f"\n{'Trait':<20} {'PLS 6204':<15} {'Mean':<15} {'Difference':<15} {'% Diff':<12} {'Status':<10}")
print("-"*80)

results = []
for trait in trait_columns:
    pls_val = pls_6204[trait]
    mean_val = trait_means[trait]
    diff = pls_val - mean_val
    
    # Calculate percent difference (handle division by zero)
    if abs(mean_val) > 1e-10:
        pct_diff = (diff / mean_val) * 100
    else:
        pct_diff = 0.0 if abs(diff) < 1e-10 else np.inf
    
    # Determine status
    if abs(pct_diff) < 10:
        status = "Similar"
    elif pct_diff > 0:
        status = "Higher"
    else:
        status = "Lower"
    
    results.append({
        'trait': trait,
        'pls_val': pls_val,
        'mean_val': mean_val,
        'diff': diff,
        'pct_diff': pct_diff,
        'status': status
    })
    
    print(f"{trait:<20} {pls_val:<15.6f} {mean_val:<15.6f} {diff:<15.6f} {pct_diff:<12.2f} {status:<10}")

print("="*80)
print("\nKEY DIFFERENCES (>50% deviation from mean):")
print("-"*80)

extreme_deviations = [r for r in results if abs(r['pct_diff']) > 50]
if extreme_deviations:
    for r in extreme_deviations:
        print(f"  {r['trait']:<20}: {r['pct_diff']:>7.1f}% {'above' if r['pct_diff'] > 0 else 'below'} mean")
else:
    print("  No extreme deviations found.")

print("\n" + "="*80)
print("SUMMARY:")
print("-"*80)
n_higher = sum(1 for r in results if r['status'] == 'Higher')
n_lower = sum(1 for r in results if r['status'] == 'Lower')
n_similar = sum(1 for r in results if r['status'] == 'Similar')

print(f"  Traits higher than mean: {n_higher}")
print(f"  Traits lower than mean:  {n_lower}")
print(f"  Traits similar to mean:  {n_similar}")
print("="*80)

# Calculate overall "distance" from mean (normalized Euclidean distance)
normalized_diff = []
for trait in trait_columns:
    pls_val = pls_6204[trait]
    mean_val = trait_means[trait]
    std_val = df[trait].std()
    if std_val > 1e-10:
        normalized_diff.append(((pls_val - mean_val) / std_val) ** 2)

distance = np.sqrt(np.sum(normalized_diff))
print(f"\nNormalized Euclidean distance from mean: {distance:.4f}")
print(f"(Distance of 0 = exactly average, >3 = unusual, >5 = very unusual)")
print("="*80)
