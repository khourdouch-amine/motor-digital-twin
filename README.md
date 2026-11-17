# Digital Twin and AI-Based Predictive Maintenance of a Three-Phase Induction Motor

## Overview

Development of a simulation-based digital twin for a three-phase induction motor, combining energy modeling, synthetic sensor data generation, vibration-based fault diagnosis, and machine learning.

## Main Objectives

- Generate synthetic motor operating scenarios.
- Process and validate sensor data.
- Estimate motor energy consumption, losses, and efficiency.
- Extract vibration indicators for bearing fault diagnosis.
- Train and evaluate machine learning models.
- Analyze model performance and robustness.

## Technologies

Python, NumPy, Pandas, Matplotlib, Scikit-learn, Joblib.

## Machine Learning

Algorithms investigated:
- Random Forest
- Logistic Regression
- Support Vector Machine (RBF)

Diagnostic features:
- RMS
- Standard deviation
- Crest factor

## Experimental Results

The reference model using three vibration features achieved:

| Metric | Score |
|---|---:|
| Accuracy | 98.94% |
| Balanced Accuracy | 98.54% |
| Precision | 99.33% |
| Recall | 97.39% |
| F1-score | 98.35% |
| ROC-AUC | 99.89% |

These results were obtained on an evaluated dataset generated from simulated scenarios. They do not represent validation on a physical industrial motor.

## Energy Modeling

The energy model estimates mechanical and electrical power, motor losses, efficiency, and cumulative energy consumption.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Project Limitations

The current study relies on synthetic sensor data and simplified simulation models. Experimental validation using real motor measurements is a necessary next step.

## Author

**Amine Khourdouch**  
Mechatronics Engineering  
CEAA – Université Ibn Tofaïl, Kenitra, Morocco
