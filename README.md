# Applying Data Science Techniques on Lung Cancer Dataset

A complete walkthrough of building and evaluating machine learning models to predict lung cancer risk from a clinical survey dataset.

---

**Project Goals**

- Clean and explore a tabular survey dataset about lung cancer risk
- Engineer features, handle class imbalance, and standardize inputs
- Train and compare multiple classifiers
- Evaluate with robust metrics beyond accuracy
- Provide a reproducible pipeline

---

**Dataset**

- File: `survey lung cancer.csv`
- Target: `LUNG_CANCER` binary, encoded as 1 cancer, 0 no cancer
- Features: mostly categorical and ordinal risk factors such as smoking status, age, symptoms
- Notes: values appear as integers 0 or 1 for most variables, some columns may require label encoding

---

**Environment**

- Python 3.x
- Core libs: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`
- Modeling: `scikit-learn`
- Optional imbalance handling: `imblearn`

---

**Workflow**

1) Import and Config  
- Set random seed for reproducibility  
- Suppress warnings for cleaner output

2) Load Data  
- `pd.read_csv('survey lung cancer.csv')`  
- Inspect shape, dtypes, head, null counts

3) Data Cleaning  
- Handle missing values, either impute mode for binary fields or drop sparse columns  
- Strip whitespace in column names  
- Verify binary columns are in {0,1} and coerce types to `int8` where possible

4) Exploratory Data Analysis  
- Class balance bar plot of target  
- Univariate distributions for key risk factors  
- Correlation heatmap for binary features using Pearson or Phi coefficient  
- Grouped bars for target vs feature to gauge signal

5) Feature Engineering  
- Ensure categorical features are encoded, use `OneHotEncoder` for multi category fields  
- Scale continuous features with `StandardScaler` inside a pipeline  
- Optional interaction features if domain justified

6) Train Test Split  
- `train_test_split(test_size=0.2, stratify=y, random_state=42)`

7) Baseline Models  
- Logistic Regression  
- Gaussian Naive Bayes  
- Decision Tree

8) Address Class Imbalance  
- If target is skewed, apply `class_weight='balanced'` for LR and Tree or use `SMOTE` within an imblearn pipeline

9) Model Selection and Tuning  
- Pipelines per model to avoid leakage  
- `StratifiedKFold` cross validation  
- `GridSearchCV` or `RandomizedSearchCV` on key hyperparameters  
  - Logistic: C, penalty, solver  
  - Tree: max_depth, min_samples_split, class_weight

10) Evaluation  
- Holdout metrics: accuracy, precision, recall, F1  
- ROC AUC and PR AUC  
- Confusion matrix  
- Calibration curve if using probabilistic outputs  
- Feature importance or coefficients for interpretability

11) Model Interpretation  
- For Logistic Regression, report top positive and negative coefficients  
- For Tree based, show feature importance bar chart

12) Save Artifacts  
- Best model with `joblib.dump`  
- Save metrics JSON and figures to `reports/`

---


**Results To Report**

- Best model type and CV F1  
- Test set accuracy, precision, recall, F1, ROC AUC  
- Confusion matrix figure and ROC curve  
- Top 10 important features

---

**Reproducibility**

- Fix random seed  
- Use pipelines to prevent leakage  
- Keep preprocessing inside CV using `GridSearchCV` with the full pipeline

---

**Limitations and Next Steps**

- Survey data may contain self report bias  
- External validation on a new cohort recommended  
- Try calibrated models and threshold tuning for recall focus  
- Evaluate additional models like Random Forest and XGBoost  
- Add SHAP for local explanations

---

**Repository Structure**

- `lung cancer.ipynb` exploratory notebook and experiments  
- `survey lung cancer.csv` raw dataset  
- `README.md` this document  
- `models/` saved best model  
- `reports/` metrics and plots
