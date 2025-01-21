#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from helper_code import *

# Function to calculate Net Benefit
def calculate_net_benefit(y_true, y_pred, threshold=0.5):
    """
    Calculate the net benefit of predictions.
    
    Parameters:
    - y_true (array-like): Ground truth binary labels (0 or 1).
    - y_pred (array-like): Predicted probabilities.
    - threshold (float): Threshold probability for classification (default is 0.5).
    
    Returns:
    - float: Net benefit score.
    """
    tp = np.sum((y_true == 1) & (y_pred >= threshold))
    fp = np.sum((y_true == 0) & (y_pred >= threshold))
    n = len(y_true)
    
    # Calculate Net Benefit
    net_benefit = (tp / n) - ((threshold / (1 - threshold)) * (fp / n))
    return net_benefit

# Function to calculate ECE (Estimated Calibration Error)
def calculate_ece(probs, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(labels[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += bin_size * np.abs(bin_acc - bin_conf) / len(probs)
    return ece

# Function to evaluate model
def evaluate_model(label_folder, output_folder):
    # Load labels and model outputs.
    _, _, label, _ = load_challenge_data(label_folder)
    patient_ids, prediction_probability, prediction_binary = load_challenge_predictions(output_folder)

    # Compute confusion matrix and metrics
    tn, fp, fn, tp = compute_confusion_matrix(label, prediction_binary)
    accuracy = compute_accuracy(tn, fp, fn, tp)

    # Additional metrics
    auc_score = roc_auc_score(label, prediction_probability)
    precision, recall, _ = precision_recall_curve(label, prediction_probability)
    auprc = auc(recall, precision)
    net_benefit = calculate_net_benefit(label, prediction_probability)
    ece = calculate_ece(prediction_probability, label)

    # Compute leaderboard score
    w_A, w_Ap, w_Nb, w_ECE = 0.3, 0.4, 0.4, -0.1
    leaderboard_score = (
        w_A * auc_score +
        w_Ap * auprc +
        w_Nb * net_benefit +
        w_ECE * ece
    )

    return {
        #'Challenge Score': challenge_score(label, prediction_probability),
        'AUC': auc_score,
        'AUPRC': auprc,
        'Net Benefit': net_benefit,
        'ECE': ece,
        'Leaderboard Score': leaderboard_score
    }

# Helper functions
def compute_confusion_matrix(labels, predictions):
    labels = np.array(labels).astype(int)
    predictions = np.array(predictions).astype(int)
    cm = np.zeros((2, 2))
    for i in range(len(labels)):
        cm[labels[i]][predictions[i]] += 1
    return cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

def compute_accuracy(tn, fp, fn, tp):
    return (tp + tn) / (tn + fp + fn + tp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance.")
    parser.add_argument("label_folder", type=str, help="Folder containing ground truth labels.")
    parser.add_argument("output_folder", type=str, help="Folder containing model predictions.")
    parser.add_argument("--output_file", type=str, help="File to save the evaluation results.")

    args = parser.parse_args()

    # Perform evaluation
    metrics = evaluate_model(args.label_folder, args.output_folder)

    # Construct output string
    output_string = '\n'.join([f"{key}: {value:.3f}" for key, value in metrics.items()])

    # Print or save the results
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output_string)
    else:
        print(output_string)
