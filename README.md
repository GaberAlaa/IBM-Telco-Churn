# Telco Customer Intelligence - AXIS ML Graduation Project

An end-to-end machine learning project on the IBM Telco Customer Churn dataset. It predicts **which
customers are likely to churn** (classification), **how much a customer is worth** (regression on
CLTV), and presents both alongside the underlying EDA in an interactive **Streamlit dashboard** with
a live, single-customer churn-risk predictor.

---

## 1. Business Problem

A telecom company wants to reduce churn and protect customer value. This project answers three
questions from the data:

1. **Which customers are likely to leave?** → Classification (target: `Churn Value`)
2. **What drives their long-term value?** → Regression (target: `CLTV`)
3. **How can the business act on this?** → A dashboard translating both into plain-language,
   retention-focused insights, plus a form to score a new/hypothetical customer in real time.

## 2. Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- **Raw file:** `Data/IBM_Telco_churn.csv` — 7,043 customers x 33 columns (demographics, account &
  service details, geography, billing, plus IBM's own `Churn Label`, `Churn Score` and `CLTV` fields).
- **Cleaned file:** `Data/Clean_IBM_Telco_churn.csv` — 7,043 rows x 21 columns. Geography,
  identifier and free-text columns (`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`,
  `Lat Long`/`Latitude`/`Longitude`, `Churn Reason`) were dropped, `Total Charges` was cast to
  numeric, and `Churn Score` was excluded from modeling entirely.

> **Why `Churn Score` is excluded:** it is IBM's own SPSS-Modeler churn prediction for each
> customer — i.e. the output of a different churn model — not a real customer attribute available
> at prediction time. Using it as a feature would be target leakage (see Section 7).

Class balance on the classification target: **~73.5% stayed / ~26.5% churned** — imbalanced on
purpose, per the assignment brief, and handled explicitly throughout (Section 5).

## 3. Repository Structure

```
Gradproject/
├── Data/
│   ├── IBM_Telco_churn.csv          # raw dataset
│   └── Clean_IBM_Telco_churn.csv    # cleaned, model-ready dataset
├── EDA.ipynb                        # Phase 1: cleaning + exploratory analysis, generates the pics
├── Hypertune.ipynb                  # Phase 2/3: hyperparameter search only (occasional, expensive)
├── Models.ipynb                     # Phase 2/3: final training with the best params found (fast, repeatable)
├── Hyper results/
│   ├── reg_results_hyper.csv        # regression search results + best params found
│   └── class_results_hyper.csv      # classification search results + best params found
├── Dashboard.py                     # Streamlit entry point (multipage app)
├── Dashboard/
│   ├── Pages/
│   │   ├── Dashboard.py             # KPI overview + churn-driver charts
│   │   ├── EDA.py                   # narrated EDA walkthrough
│   │   ├── Classification.py        # classifier comparison + chosen model
│   │   ├── Reggresion.py            # regressor comparison + chosen model
│   │   └── predict.py               # live single-customer churn predictor
│   └── Resources/
│       ├── regression_results.csv       # final regression comparison table (feeds the dashboard)
│       ├── classification_results.csv   # final classification comparison table (feeds the dashboard)
│       ├── Model/Randf_clf.pkl          # the saved, trained classification pipeline
│       └── pics/graph_1.png ... graph_8.png   # EDA charts referenced by Dashboard/Pages/EDA.py
├── config.py                        # single source of truth for paths, column groups, targets
└── README.md
```

## 4. Configuration (`config.py`)

Every script/notebook imports its paths and column groups from here, so preprocessing stays
identical across the whole project:

| Group                   | Columns                                                                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NUMERIC_COLS`          | Tenure Months, Monthly Charges, Total Charges                                                                                                               |
| `BINARY_COLS`           | Gender, Senior Citizen, Partner, Dependents, Phone Service, Paperless Billing                                                                               |
| `MULTI_CAT_COLS`        | Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies, Contract, Payment Method |
| `REGRESSION_TARGET`     | `CLTV`                                                                                                                                                      |
| `CLASSIFICATION_TARGET` | `Churn Value`                                                                                                                                               |

Preprocessing (both notebooks and the dashboard's saved model) uses one `ColumnTransformer`:
`StandardScaler` on numeric columns, `OneHotEncoder(drop="if_binary")` on binary columns, and plain
`OneHotEncoder` on multi-category columns.

## 5. Methodology

### 5.1 EDA & Cleaning (`EDA.ipynb`)

Missing values (`Total Charges` had a small number of blanks), data types, and outlier checks
(boxplots on Tenure/Monthly/Total Charges — none required removal) were handled here, along with
the 8 charts saved to `Dashboard/Resources/pics/` and narrated on the dashboard's EDA page. Headline
findings: contract length is the single strongest churn signal (month-to-month churns at ~43% vs.
~3% on two-year contracts), churners pay more on average (~$80 vs. ~$64/month), and electronic-check
payers churn roughly 3x more often than customers on automatic payment.

### 5.2 Regression — predicting CLTV

**Features:** all numeric + binary + multi-category columns (`Churn Value` is deliberately excluded
— it isn't known ahead of time for a customer who hasn't churned yet, so including it would leak
information not actually available at prediction time).

**Models compared:** Linear, Polynomial, Lasso, Ridge, Random Forest Regressor, Gradient Boosting
Regressor.

### 5.3 Classification — predicting churn

**Features:** all numeric + binary + multi-category columns **plus `CLTV`** — unlike `Churn Value`,
CLTV is a value the company already has on hand for a customer before they churn, so it's a
legitimate predictive feature here.

**Models compared:** Logistic Regression, KNN, SVM, Decision Tree, Naive Bayes, Random Forest,
XGBoost.

**Imbalance handling:** `class_weight="balanced"` on every model that supports it (Logistic
Regression, SVM, Decision Tree, Random Forest). Models with no such parameter get an equivalent
fix — XGBoost via `scale_pos_weight`, Gradient-Boosting-style models via `sample_weight`. KNN and
Naive Bayes have no imbalance mechanism in scikit-learn at all, which is called out explicitly
rather than left unaddressed — Naive Bayes ends up with the highest raw recall of the batch (0.83)
largely as a side effect of this, at the cost of much lower precision.

**Evaluation:** Accuracy, Precision, Recall, F1 and ROC-AUC are all reported — **not accuracy
alone**, since on a 73/25 split a model that always predicts "stays" would already score ~73%
accuracy while catching zero churners.

### 5.4 Hyperparameter tuning — split into two notebooks by design

- **`Hypertune.ipynb`** runs the actual search (`GridSearchCV` for cheap/low-dimensional models
  like Ridge/Lasso/Logistic Regression, `RandomizedSearchCV` for the slower ones — SVM, Random
  Forest, Gradient Boosting, XGBoost — where an exhaustive grid was too slow to cover a useful
  range) and records the winning parameters to `Hyper results/*.csv`.
- **`Models.ipynb`** is the fast path: it trains each model once with the best parameters already
  found, evaluates on the held-out test set, saves the comparison tables to
  `Dashboard/Resources/*.csv`, and persists the chosen classifier to
  `Dashboard/Resources/Model/Randf_clf.pkl`.

  This split matters in practice, not just in principle: `Hypertune.ipynb` runs the expensive
  search only when you actually want to re-tune (e.g. after a meaningful new batch of data or a
  drop in monitored performance); `Models.ipynb` is what you'd re-run on every new batch, since a
  plain `.fit()` with fixed parameters is far cheaper than repeating a multi-fold search every time.

## 6. Results

### Regression (target: CLTV) — `Dashboard/Resources/regression_results.csv`

| Model                       | RMSE    | MAE    | R2    |
| --------------------------- | ------- | ------ | ----- |
| Random Forest Regressor     | 1025.55 | 872.92 | 0.226 |
| Gradient Boosting Regressor | 1026.09 | 873.30 | 0.225 |
| Polynomial Regression       | 1038.58 | 883.20 | 0.206 |
| Lasso Regression            | 1061.70 | 898.20 | 0.170 |
| Ridge Regression            | 1062.02 | 898.43 | 0.170 |
| Linear Regression           | 1062.44 | 897.94 | 0.169 |

**Chosen model: Random Forest Regressor** — best on every metric. R2 tops out around 0.23,
which reflects a real ceiling in the available features rather than under-tuning: CLTV appears to
depend on inputs (e.g. IBM's own predictive scoring, or deeper account history) not present in this
dataset. See Section 7 for why Random Forest Regressor underperforms even the plain linear models here.

### Classification (target: Churn Value) — `Dashboard/Resources/classification_results.csv`

| Model               | Accuracy  | Precision | Recall    | F1        | ROC-AUC |
| ------------------- | --------- | --------- | --------- | --------- | ------- |
| **Random Forest**   | 0.774     | 0.551     | **0.799** | **0.652** | 0.856   |
| XG Boosting         | 0.764     | 0.538     | 0.802     | 0.644     | 0.854   |
| SVM                 | 0.758     | 0.530     | 0.786     | 0.633     | 0.839   |
| Decision Tree       | 0.752     | 0.522     | 0.770     | 0.622     | 0.836   |
| Logistic Regression | 0.744     | 0.512     | 0.783     | 0.619     | 0.848   |
| Naive Bayes         | 0.708     | 0.471     | 0.826     | 0.600     | 0.817   |
| KNN                 | **0.781** | **0.587** | 0.596     | 0.592     | 0.823   |

**Chosen model: Random Forest** — highest F1 (best balance of catching churners vs. false alarms)
and the second-highest ROC-AUC, with recall of ~0.80 (catches roughly 4 in 5 actual churners). This
is the model saved to `Dashboard/Resources/Model/Randf_clf.pkl` and used by the live predictor page.
XGBoost is a close second on nearly every metric and is worth revisiting with more tuning budget.

## 7. Setup & Installation

```bash
git clone <repo-url>
cd Gradproject
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn scipy xgboost joblib streamlit plotly seaborn matplotlib jupyter
```

## 8. How to Run

**Reproduce the full pipeline from scratch**, in order:

1. `EDA.ipynb` — cleans the raw data and regenerates the EDA charts.
2. `Hypertune.ipynb` — runs the hyperparameter search (slow; only needed when re-tuning).
3. `Models.ipynb` — trains final models with the best known parameters, writes the results CSVs
   and saves the classifier `.pkl` used by the dashboard.

**Launch the dashboard** (after step 3, or using the results/model already checked in):

```bash
streamlit run Dashboard.py
```

This opens a 5-page app: **Dashboard** (KPI overview + churn-driver charts), **EDA** (narrated
walkthrough), **Classification** (model comparison + chosen model), **Regression** (model comparison

- chosen model), and **Predict** (live churn-risk scoring for a hypothetical customer).

## 9. Acknowledgments

Dataset: [IBM Telco Customer Churn, via Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset),
