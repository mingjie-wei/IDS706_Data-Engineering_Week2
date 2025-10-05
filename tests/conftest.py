import pandas as pd
import pytest


@pytest.fixture
def tiny_df():
    return pd.DataFrame(
        {
            "Customer_ID": ["A", "B", "C", "D"],
            "Age": [25, 40, 35, 29],
            "Gender": ["Female", "Male", "Female", "Female"],
            "Purchase_Category": [
                "Electronics",
                "Clothing",
                "Electronics",
                "Food & Beverages",
            ],
            "Purchase_Amount": ["$100.00 ", "$250.00 ", "$150.00 ", "$50.00 "],
            "Frequency_of_Purchase": [2, 6, 4, 3],
            "Average_Order_Value": [50.0, 125.0, 75.0, 50.0],
            "Return_Rate": [1, 0, 0, 2],
            "Customer_Satisfaction": [6, 8, 9, 5],
            "Engagement_with_Ads": ["High", "Low", "None", "Medium"],
            "Device_Used_for_Shopping": ["Mobile", "Desktop", "Tablet", "Mobile"],
            "Payment_Method": ["Credit Card", "PayPal", "Credit Card", "Debit Card"],
            "Time_of_Purchase": ["Morning", "Evening", "Afternoon", "Morning"],
            "Discount_Used": [True, False, True, False],
            "Customer_Loyalty_Program_Member": [False, True, True, False],
            "Purchase_Intent": ["Need-based", "Wants-based", "Impulsive", "Need-based"],
            "Shipping_Preference": ["Standard", "Express", "No Preference", "Standard"],
            "Time_to_Decision": [2, 6, 3, 10],
        }
    )
