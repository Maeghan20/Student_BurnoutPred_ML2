# Student Burnout — Regularized Regression Lab 2

## MAI511-2 Advanced Machine Learning

**Topic:** Regularized Regression using Lasso and Ridge with Cross-Validation and Grid Search

This project implements the complete lab exercise using the supplied `student_burnout.csv` dataset.

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

### Important preprocessing decisions
- `student_id` is excluded because it is only an identifier.
- `high_burnout` is excluded because it is derived from `burnout_score`; including it would cause target leakage.
- Missing numeric values: median imputation.
- Missing categorical values: most-frequent imputation.
- `gender`: one-hot encoding.
- Numeric features: StandardScaler.
- All preprocessing is placed inside sklearn Pipelines/ColumnTransformer.

### Files
| File | Purpose |
|---|---|
| `student_burnout.csv` | Supplied dataset |
| `Student_Burnout_Regularized_Regression_Lab2.ipynb` | Full lab notebook with explanations |
| `app.py` | Sleek Streamlit dashboard |
| `requirements.txt` | Python dependencies |
| `Viva_Cheat_Sheet.txt` | Viva-focused explanations |
| `Lab2_Report.md` | Submission-ready write-up |
| `.gitignore` | Prevents virtual environments/cache files from being committed |

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not on PATH:

```powershell
python -m streamlit run app.py
```

or use the Python launcher/executable available on your machine.

## Streamlit deployment

Push this folder to GitHub. On Streamlit Community Cloud:
- Repository: your GitHub repository
- Branch: `main`
- Main file: `app.py`

The CSV must remain in the repository root because `app.py` loads `student_burnout.csv`.

## Submission naming

The lab sheet says the submitted file should follow a pattern containing the student's name, last three register-number digits, and program number. Replace the placeholder with your actual required naming convention before uploading to Google Classroom.
