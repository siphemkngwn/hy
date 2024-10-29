# Pediatric Sepsis Data Challenge: In-Hospital Mortality Prediction Task

<!-- Brief introduction to the challenge and its objectives -->
Welcome to the Pediatric Sepsis Data Challenge! This challenge focuses on predicting in-hospital mortality for pediatric sepsis cases using a synthetic dataset derived from real-world data. The ultimate goal is to improve early detection models for better resource allocation and clinical outcomes in low-resource healthcare settings.

## Objective

<!-- State the primary task for participants -->
Develop an open-source algorithm to predict in-hospital mortality among children with sepsis. This algorithm should be trained solely on the provided dataset, using any or all variables available within.

## Contents

<!-- Table of contents for easy navigation in a Markdown file -->
1. [Data and Code Requirements](#1-data-and-code-requirements)
2. [Submission Guidelines and Limits](#2-submission-guidelines-and-limits)
3. [Submission Components](#3-submission-components)
4. [Testing and Evaluation Criteria](#4-testing-and-evaluation-criteria)
5. [Model Preferences](#5-model-preferences)
6. [Final Instructions](#6-final-instructions)

---

### 1. Data and Code Requirements

#### Dataset

- **Provided Dataset**: Synthetic data derived from real hospital data from Uganda.
- **Feature Constraints**: Your algorithm should exclusively use the provided dataset variables for predictions.

#### Submission Requirements

- **Code and Model**: Submit both:
  - **Training Code**: All scripts and code required for training the model.
  - **Trained Model**: The model file generated from your code.
- **Language**: Submissions must be in Python; however, R, Julia, and MATLAB submissions are also accepted. Python is recommended to facilitate baseline comparisons.

#### Code Validity

- **Environment**: Code will run in a containerized setup.
- **Execution Time**: Maximum 24 hours for training, with 8 hours allocated for validation and testing.
- **Autonomous Execution**: Ensure your code can execute end-to-end without manual intervention.
  - **Dependencies**: List all dependencies in `requirements.txt` or a compatible environment configuration file.
  - **Preprocessing**: Include any data preprocessing or transformations directly within the submitted code.

---

### 2. Submission Guidelines and Limits

#### Submission Limit

- Each team may submit code up to **3 times** throughout the challenge.

#### Evaluation

- Each submission will be assessed on a hidden evaluation set to ensure unbiased scoring.
- Only the **final model from each training phase** will be evaluated for the official score.

#### Repository Security

- Teams are expected to maintain their code in **private repositories** during the challenge to ensure fairness.

#### Post-Challenge Public Release

<!-- Explain the requirements for the public release of solutions after the challenge concludes -->
Upon completion, all final solutions must be shared publicly (e.g., GitHub) to promote reproducibility and transparency.

**Public Release Requirements**:
- Complete source code and trained models.
- Detailed README file with instructions for replication.
- An open-source license (e.g., MIT, BSD) specifying usage and redistribution rights.

---

### 3. Submission Components

Each submission should include:

#### Source Code
- Scripts for **Data Preprocessing**, **Model Training**, and **Prediction on Test Data**.

#### Documentation
- A comprehensive README file detailing:
  - **How to run the code**.
  - Any specific assumptions or variable handling.
  - Unique features of your model or algorithm.

#### Environment Setup
- **requirements.txt** for listing dependencies.
- (Optional) **Dockerfile** for any special environment configurations.

#### Model File
- Trained model saved in a standard format (e.g., `.pkl` for scikit-learn or `.h5` for TensorFlow/Keras).

---

### 4. Testing and Evaluation Criteria (Tentative)

<!-- Details on how submissions will be evaluated based on several key metrics -->
Your model will be evaluated on the following metrics:

1. **True Positive Rate (TPR) at False Positive Rate (FPR) ≤ 0.20**: Prioritizes high detection rates for critical cases while minimizing false positives.
2. **Positive Predictive Value (PPV)**: Ensures predictions are accurate to limit unnecessary interventions in resource-limited settings.
3. **Area Under the ROC Curve (AUC-ROC)**: A secondary metric to measure general performance across thresholds.
4. **Balanced Accuracy**: Averages recall for both classes, addressing class imbalance.
5. **F1-Score**: Balances precision and recall, providing a stable metric under class imbalance conditions.

---

### 5. Model Preferences

<!-- Guidance on preferred model characteristics given the constraints and goals of the challenge -->
The challenge favors models that:

- **Optimize Predictive Power**: High predictive accuracy with minimal variable dependency.
- **Consider Resource Constraints**: Prefer parsimonious models suited for environments with limited computational and clinical resources.

---

### 6. Final Instructions

#### Autonomous Execution
- Ensure all components of your submission run autonomously from start to finish in a **cloud-based container**.

#### Leaderboard
- Scores will be updated on the leaderboard based on the **best score achieved**.

#### Open-Source Compliance
- Ensure that your final submission is properly documented and made available publicly.

---

We are excited to see your innovative solutions aimed at improving pediatric sepsis outcomes in resource-constrained settings!

