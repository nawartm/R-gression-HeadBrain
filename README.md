# Regression Project: Predicting Brain Weight and Advertising Sales

> 🧠 *Estimate brain weight based on skull volume*  
> 📺 *Predict sales based on advertising budgets (TV, radio, newspaper)*

This Jupyter notebook (`Regression.ipynb`) presents **two complete case studies in Machine Learning regression**:
1. **Simple regression**: Predict human brain weight from skull volume.
2. **Multiple regression**: Predict product sales based on advertising spend across different media.
3. **Model comparison**: Linear regression, decision trees, and random forests.

---

## Project Objectives

### 1. Simple Regression — HeadBrain Dataset
> *Can we estimate a person’s brain weight simply by measuring their skull volume?*

- ✅ Clean data (missing values, duplicates, outliers).
- ✅ Visualize the relationship between skull volume and brain weight.
- ✅ Build a simple linear regression model.
- ✅ Evaluate model performance using key metrics (MAE, MSE, R², etc.).
- ✅ Compare results with and without outliers.

### 2. Multiple Regression — Advertising Dataset
> *What is the impact of advertising budgets (TV, radio, newspaper) on sales?*

- ✅ Explore correlations between variables.
- ✅ Build two models:
  - Full model: `Sales = f(TV, radio, newspaper)`
  - Reduced model: `Sales = f(TV, radio)`
- ✅ Compare performances to determine whether the “newspaper” variable adds value.
- ✅ Test advanced models: **Decision Trees** and **Random Forests**.

---

## Target Audience

| Audience | What They Will Find |
|----------|----------------------|
| **Students in Data / Stats / ML** | A complete, step-by-step tutorial with executable code, visualizations, and clear explanations — perfect for learning or review. |
| **Teachers / Trainers** | A ready-to-use pedagogical resource to illustrate linear regression, data cleaning, and model evaluation. |
| **Junior Data Scientists** | A practical example of a modeling pipeline: from exploration to model comparison. |
| **Non-Technical Readers (Managers, Curious)** | Simple explanations, intuitive visualizations, and concrete results to understand how Machine Learning can answer scientific or business questions. |

---

## Technical Steps Implemented

### Data Exploration & Cleaning
- Load data using `pandas`
- Detect duplicates and missing values
- Visualize outliers using `boxplots` (IQR method)
- Remove outliers to improve model quality

### Visual & Statistical Analysis
- Scatter plots to observe relationships
- Correlation matrices using `seaborn.heatmap`
- Pairplots to visualize distributions and pairwise relationships across all variables

### Modeling
#### Simple & Multiple Linear Regression
- Use `sklearn.linear_model.LinearRegression`
- Split data: `train_test_split`
- Compute coefficients (a, b) and interpret them
- Predict on test set

#### Advanced Models
- **Decision Tree**: `DecisionTreeRegressor`
- **Random Forest**: `RandomForestRegressor`

### Model Evaluation
Metrics computed for each model:
- **MAE** (Mean Absolute Error) → Average absolute error
- **MSE** (Mean Squared Error) → Penalizes large errors
- **RMSE** (Root Mean Squared Error) → Interpretable in target unit
- **R²** (Coefficient of Determination) → % of variance explained (0 to 1, 1 = perfect)
- **R** (Square root of R²) → Predictive correlation

---

## Technologies & Libraries Used

```python
import pandas as pd        # Data manipulation
import numpy as np         # Numerical computations
import seaborn as sns      # Statistical visualizations
import matplotlib.pyplot as plt  # Plotting

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
