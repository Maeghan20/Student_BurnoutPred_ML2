# Student Burnout — Regularized Regression Lab 2

**Topic:** Regularized Regression using Lasso and Ridge with Cross-Validation and Grid Search

This project implements the complete lab exercise using the `student_burnout.csv` dataset.

### Dataset
- 2,000 records
- 17 columns
- Continuous target: `burnout_score`
- Categorical feature: `gender`
- Missing values are present in several numeric columns
- No duplicate rows

### Required models
1. **Linear Regression** — baseline
2. **Lasso Regression** — L1 regularization + 5-fold GridSearchCV
3. **Ridge Regression** — L2 regularization + 5-fold GridSearchCV

### Required evaluation
- MAE
- MSE
- RMSE
- R²

### Required visualizations
- Feature vs target relationships
- Actual vs Predicted for Lasso and Ridge
- Coefficient comparison
- Cross-validation RMSE vs alpha

### Self-learning / additional exploration
An extra **Polynomial Ridge** experiment is included. It creates pairwise interaction features and then applies Ridge regularization. The goal is to investigate whether the additional interactions can reduce test error.

### Preprocessing 
- `student_id` is excluded because it is only an identifier.
- `high_burnout` is excluded because it is derived from `burnout_score`; including it would cause target leakage.
- Missing numeric values: median imputation.
- Missing categorical values: most-frequent imputation.
- `gender`: one-hot encoding.
- Numeric features: StandardScaler.
- All preprocessing is placed inside sklearn Pipelines/ColumnTransformer.

