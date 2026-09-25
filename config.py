
DATA_PATH = 'IBM Telco/IBM_Telco_churn.csv'
CLEAN_DATA_PATH = 'IBM Telco/Clean_IBM_Telco_churn.csv'
RANDOM_STATE = 42

binary_cols = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
    "Paperless Billing", "Churn Label",
]
multi_cat_cols = [
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Payment Method",
]
