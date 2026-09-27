
DATA_PATH = 'IBM Telco/IBM_Telco_churn.csv'
CLEAN_DATA_PATH = 'IBM Telco/Clean_IBM_Telco_churn.csv'
DASHBOARD_RESOURCES_PATH = 'Dashboard/Resources'
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
