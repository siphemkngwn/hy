#Pediatric Sepsis Data Challenge: In-Hospital Mortality Prediction Task

Welcome to the Pediatric Sepsis Data Challenge! This competition aims to create models for predicting in-hospital mortality for pediatric sepsis cases, with the goal of enhancing early detection and improving resource allocation in low-resource healthcare settings.

Objective

The task is to develop an open-source algorithm predicting in-hospital mortality among children with sepsis, using the provided synthetic dataset. The project builds on existing work available in the 2023 Pediatric Sepsis Challenge repository, which participants may reference and extend as needed.

Contents

1. Data and Code Requirements
2. Submission Guidelines and Limits
3. Submission Components
4. Testing and Evaluation Criteria
5. Model Preferences
6. Final Instructions
1. Data and Code Requirements
Dataset

Provided Dataset: A synthetic dataset derived from real-world hospital data in Uganda, found in the project repository.
Feature Constraints: Use only the provided dataset’s variables for training and prediction.
Submission Requirements

Code and Model: Submit both:
Training Code: All scripts and code required to train the model.
Trained Model: The model file generated from your code.
Language: Submissions must be in Python (preferred for baseline comparisons); R, Julia, and MATLAB are also accepted.
Code Validity

Environment: Code will be executed in a containerized environment.
Execution Time: 24 hours max for training and 8 hours max for validation and testing.
Autonomous Execution: Ensure your code runs end-to-end without manual intervention.
Dependencies: List all dependencies in requirements.txt or use an environment configuration file.
Preprocessing: Include all data preprocessing steps within your code submission.
2. Submission Guidelines and Limits
Submission Limit: Each team is allowed 3 submissions during the competition.
Evaluation: Submissions will be evaluated on a hidden test set for unbiased scoring.
Only the final trained model from each submission will be scored.
Repository Security: Teams should keep their code in private repositories to ensure fair competition.
Post-Challenge Public Release

After the challenge, final solutions must be shared publicly (e.g., GitHub) to promote reproducibility and transparency.

Public Release Requirements:
Complete source code and trained models.
Detailed README with replication instructions.
Open-source license (e.g., MIT, BSD) defining usage and redistribution rights.
3. Submission Components
Each submission should include:

Source Code:
Scripts for Data Preprocessing, Model Training, and Prediction on Test Data.
Documentation:
A README file detailing:
Instructions for running the code.
Any assumptions or special handling of variables.
Unique features of your model.
Environment Setup:
requirements.txt listing dependencies.
(Optional) Dockerfile for specific environment configurations.
Model File:
Trained model saved in a standard format (e.g., .pkl for scikit-learn or .h5 for TensorFlow/Keras).
For guidance, refer to The 2023 Pediatric Sepsis Challenge repository as a reference framework.

4. Testing and Evaluation Criteria
Models will be evaluated on these criteria:

True Positive Rate (TPR) at False Positive Rate (FPR) ≤ 0.20: Focus on detecting critical cases with minimal false positives.
Positive Predictive Value (PPV): High accuracy to limit unnecessary interventions in resource-limited settings.
Area Under the ROC Curve (AUC-ROC): Secondary metric to assess general performance.
Balanced Accuracy: Addresses class imbalance by averaging recall for both classes.
F1-Score: Balances precision and recall, providing a stable measure under class imbalance.
5. Model Preferences
Preferred model characteristics:

Resource Efficiency: High predictive power with minimal data dependency.
Suitability for Low-Resource Environments: Parsimonious models are encouraged.
6. Final Instructions
Autonomous Execution: Ensure all components of your submission can execute independently in a cloud-based container.
Leaderboard: Scores will be updated on the leaderboard based on the best score achieved per team.
Open-Source Compliance: Final solutions should be documented and made available in public repositories like GitHub.
We look forward to your contributions and innovative solutions for pediatric sepsis care in resource-constrained environments!