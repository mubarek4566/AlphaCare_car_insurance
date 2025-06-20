# AlphaCare Car Insurance Risk Analytics 
This repository showcases predictive models for optimizing car insurance marketing and risk management at AlphaCare Insurance Solutions (ACIS). Using historical claim data, it identifies trends, low-risk clients, and opportunities to reduce premiums, enhancing customer acquisition and retention through data-driven strategies.

## Objectives
- Identify low-risk customers for premium optimization
- Conduct A/B hypothesis testing
- Build predictive models for premium and claim prediction
- Provide actionable insights for marketing decisions

## Tech Stack
- Python
- Pandas, Scikit-learn, Statsmodels, Seaborn, Matplotlib
- Git, GitHub
- GitHub Actions (CI/CD)

## 🗃️ Data Version Control (DVC)

To ensure reproducibility and compliance in a regulated industry like insurance, DVC is used to version-control datasets and pipeline artifacts.

### Steps to Reproduce:
1. Install DVC:
   ```bash
   pip install dvc

Initialize DVC:
    dvc init

Add Local Remote Storage:
    mkdir -p ./dvc-storage
    dvc remote add -d localstorage ./dvc-storage

Track and Version Raw Dataset:
    dvc add data/raw/insurance_data.csv
    git add data/raw/insurance_data.csv.dvc .gitignore
    git commit -m "Track raw dataset with DVC"

Push Dataset to Local Remote:
    dvc push   

 # ✅ Model implementation:

## Data Preparation:

Handles missing data with median imputation for numerical features and 'Unknown' for categorical features

Creates new features like vehicle age, coverage ratio, and power-to-capacity ratio

Converts dates to datetime and extracts temporal features

## Model Building:

💡 Claim severity model (Random Forest Regressor) trained only on policies with claims

💡 Claim probability model (Random Forest Classifier) trained on all policies

💡 Uses pipelines with preprocessing for both numerical and categorical features

## Evaluation:

💡 For regression: RMSE and R-squared

💡 For classification: AUC and accuracy

## Risk-Based Pricing:

💡 Implements the formula: Premium = (Predicted Probability * Predicted Severity) * (1 + Expense + Profit)

## Modularity:

💡 Models can be saved and loaded for later use

💡 Clear separation of concerns with different functions for each task