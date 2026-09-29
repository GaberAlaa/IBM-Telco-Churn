
DATA_PATH = 'Data/IBM_Telco_churn.csv'
CLEAN_DATA_PATH = 'Data/Clean_IBM_Telco_churn.csv'

CLASSIFICATION_RESULTS_PATH = 'Dashboard/Resources/classification_results.csv'
REGRESSION_RESULTS_PATH = 'Dashboard/Resources/regression_results.csv'

DASHBOARD_RESOURCES_PATH = 'Dashboard/Resources'
DASHBOARD_Pics_PATH = 'Dashboard/Resources/Pics'

CLASSIFICATION_MODEL_PATH = "Dashboard/Resources/Model/Randf_clf.pkl"

RANDOM_STATE = 42

NUMERIC_COLS = ["Tenure Months", "Monthly Charges", "Total Charges"]

BINARY_COLS = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
    "Paperless Billing"
]

MULTI_CAT_COLS = [
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Payment Method"
]

REGRESSION_TARGET = "CLTV"
CLASSIFICATION_TARGET = "Churn Value"
