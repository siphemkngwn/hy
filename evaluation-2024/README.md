# Scoring code for The 2024 Pediatric Sepsis Challenge

This repository contains the Python evaluation code for The 2024 Pediatric Sepsis Challenge.

The `evaluate_model` script evaluates the outputs of your models using the evaluation metric that is described on the [webpage](https://sepsis.ubc.ca/research/current-research-projects/pediatric-sepsis-data-challenge) for the 2024 Challenge. This script reports multiple evaluation metrics, so check the scoring section of the webpage to see how we evaluate and rank your models.

## Python

You can run the Python evaluation code by installing the NumPy package and running the following command in your terminal:

    python evaluate_model.py labels outputs scores.csv

where

- `labels` (input; required) is a folder with labels for the data;
- `outputs` (input; required) is a folder containing files with your model's outputs for the data; and
- `scores.csv` (output; optional) is a collection of scores for your model.


## Troubleshooting

Unable to run this code with your code? Try one of the example codes on the training data. Unable to install or run Python? Try [Python](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/products/individual), or your package manager.

## How do I learn more?

Please see the [Challenge website](https://sepsis.ubc.ca/research/current-research-projects/pediatric-sepsis-data-challenge) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/g/2024-pediatric-sepsis-data-challenge).

