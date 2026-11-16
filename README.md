# ⚡ VOLTEX – EV Battery Health Monitoring & Remaining Useful Life Prediction

**An AI-powered battery analytics system for monitoring Lithium-Ion battery health, analyzing degradation patterns, and estimating Remaining Useful Life (RUL).**

VOLTEX is a Python-based project designed to analyze Lithium-Ion battery aging data and provide insights into battery health through an interactive Streamlit dashboard. The system uses historical battery measurements to calculate State of Health (SOH), evaluate machine learning models, visualize degradation trends, and estimate the battery's remaining useful life.

---

## 📌 Project Overview

Lithium-Ion batteries gradually lose their capacity with repeated charge and discharge cycles. Monitoring this degradation is important for understanding battery performance and estimating when a battery may reach its end-of-life threshold.

VOLTEX addresses this problem through:

- Battery health monitoring using State of Health (SOH).
- Machine learning-based SOH prediction.
- Battery degradation trend analysis.
- Experimental Remaining Useful Life (RUL) estimation.
- Interactive visualizations and battery-level comparisons.
- Model performance evaluation using multiple metrics.

The project uses the **NASA Prognostics Center of Excellence (PCoE) Lithium-Ion Battery Aging Dataset**.

---

## 🎯 Project Objectives

1. Analyze Lithium-Ion battery degradation patterns.
2. Calculate battery State of Health (SOH) from discharge capacity.
3. Develop and evaluate machine learning models for SOH prediction.
4. Compare model performance using MAE, RMSE, and R².
5. Estimate Remaining Useful Life using recent SOH degradation trends.
6. Build an interactive dashboard for battery health analysis.

---

## ✨ Key Features

### 1. Battery Diagnostic Dashboard
- Displays the selected battery's latest SOH.
- Provides battery health status.
- Visualizes SOH degradation over discharge cycles.
- Displays battery capacity trends.
- Provides an overview of battery condition.

### 2. Battery Explorer
- Compare SOH trends across multiple battery units.
- Analyze differences in battery degradation.
- Explore battery performance over discharge cycles.

### 3. Machine Learning Model Performance
- Evaluate different regression algorithms.
- Compare prediction performance using MAE, RMSE, and R².
- Display model evaluation results.
- Identify the best-performing model based on evaluation metrics.

### 4. Experimental RUL Estimation
VOLTEX includes an experimental Remaining Useful Life estimation module.

The module:
- Uses recent SOH observations for the selected battery.
- Fits a Linear Regression model to the latest 60 SOH records.
- Estimates the degradation rate.
- Projects the future SOH trend.
- Estimates the cycle at which SOH reaches the 70% reference threshold.
- Calculates the estimated remaining discharge cycles.

**Important:** RUL is currently a trend-based experimental estimation, not a separately trained or validated machine learning RUL model. Its predictions should not be interpreted as guaranteed battery lifetime.

### 5. Data Laboratory
- Displays dataset records.
- Provides dataset information and data types.
- Shows missing-value statistics.
- Allows users to download the dataset as a CSV file.

---

## 🧠 Methodology

The project follows a structured machine learning workflow.

1. **Data Collection:** Obtain Lithium-Ion battery aging measurements from the NASA PCoE dataset.
2. **Data Preprocessing:** Load, organize, and prepare battery measurements.
3. **Feature Engineering:** Prepare relevant battery features for model training.
4. **SOH Calculation:** Calculate State of Health using discharge capacity.
5. **Model Training and Evaluation:** Train regression models and evaluate their performance.
6. **Degradation Analysis:** Examine SOH trends across discharge cycles.
7. **RUL Estimation:** Apply linear trend extrapolation to estimate the remaining cycles until the reference EOL threshold.
8. **Dashboard Development:** Present analytical results through an interactive Streamlit application.

---

## 📊 State of Health (SOH)

State of Health represents the remaining battery capacity relative to its initial observed discharge capacity.

The project calculates SOH using:

\[
SOH(\%) = \frac{Q_t}{Q_0}\times 100
\]

Where:

- \(Q_t\) = Discharge capacity at the current cycle.
- \(Q_0\) = Discharge capacity at the first observed cycle.

A 70% SOH threshold is used as a project reference for end-of-life estimation. It is not a universal battery safety limit.

---

## 🤖 Machine Learning Models

The project evaluates the following regression algorithms:

- Extra Trees Regressor
- Ridge Regression
- Random Forest Regressor

Model performance is evaluated using Leave-One-Battery-Out Cross-Validation (LOBO-CV), where one battery is held out for evaluation while the remaining batteries are used for training.

### Model Evaluation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Extra Trees | 2.6213 | 3.8481 | 0.8759 |
| Ridge Regression | 3.2859 | 3.8362 | 0.8766 |
| Random Forest | 3.4927 | 4.7298 | 0.8125 |

**Evaluation observations:**

- Extra Trees achieved the lowest MAE.
- Ridge Regression achieved slightly lower RMSE and higher R².
- Extra Trees provides the lowest average absolute prediction error among the evaluated models.

These results represent the reported cross-validation experiment and should not be interpreted as guaranteed performance on unseen battery populations.

---

## ⏳ Remaining Useful Life (RUL) Estimation

RUL represents the estimated number of remaining discharge cycles before a battery reaches a defined end-of-life threshold.

VOLTEX uses a simple Linear Regression-based extrapolation method:

- Recent 60 SOH observations are selected.
- A regression line is fitted against discharge cycles.
- The estimated degradation slope is calculated.
- The current SOH and degradation slope are used to project the cycle at which SOH reaches 70%.

### RUL Estimation Configuration

| Parameter | Configuration |
|---|---|
| Method | Linear Regression |
| Historical Window | 60 cycles |
| EOL Reference Threshold | 70% SOH |
| Input | Original SOH observations |
| Output | Estimated EOL cycle and RUL cycles |

### Backtesting Experiment

Different historical windows and SOH processing approaches were compared using available battery data.

The lowest overall
