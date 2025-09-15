import pandas as pd
import numpy as np

NUMERIC_CANDIDATES = [
    "Age", "Frequency_of_Purchase", "Average_Order_Value", "Return_Rate",
    "Customer_Satisfaction", "Time_to_Decision",
]

CATEGORICAL_CANDIDATES = [
    "Gender", "Income_Level", "Marital_Status", "Education_Level",
    "Occupation", "Purchase_Category", "Engagement_with_Ads",
    "Device_Used_for_Shopping", "Payment_Method", "Time_of_Purchase",
    "Discount_Used", "Customer_Loyalty_Program_Member", "Purchase_Intent",
    "Shipping_Preference",
]


def add_high_satisfaction_label(df, threshold=7):

    if "Customer_Satisfaction" not in df.columns:
        raise KeyError("'Customer_Satisfaction' not in DataFrame")
    y = (df["Customer_Satisfaction"] >= threshold).astype(int)
    y.name = "High_Satisfaction"

    return y


def select_and_encode_features(df):

    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    cats = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]

    X = df[numeric + cats].copy()
    X = pd.get_dummies(X, columns=cats, drop_first=True, dtype=np.int8)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, numeric
