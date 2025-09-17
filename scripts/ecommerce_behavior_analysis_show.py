# Ecommerce Consumer Behavior Analysis

# ======= Environment Verification =======

# Testing Kernel and Environment
import os
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
print("Python version:", sys.version)
print("Python path:", sys.executable)

# Testing essential package availability
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("All data science packages available!")
    print("pandas version:", pd.__version__)
except ImportError as e:
    print("Package import failed:", e)


# ======= Data Loading =======

def get_data_path():

    data_file_name = 'Ecommerce_Consumer_Behavior_Analysis_Data.csv'

    if os.getenv('RUN_ENVIRONMENT') == 'docker':
        return f'/app/data/{data_file_name}'
    else:
        # Default to local path for development
        return f'../data/{data_file_name}'


def save_plot(plot, filename):

    if os.getenv("IS_DOCKER_CONTAINER") == "true":
        output_dir = "/app/scripts/images"
    else:
        output_dir = "images"

    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, filename)

    plot.savefig(full_path)
    print(f"Plot saved to {full_path}")


data_path = get_data_path()

if not os.path.exists(data_path):
    print(f"Error: Unable to locate data file at {data_path}")
    print("Current working directory:", os.getcwd())
    exit(1)

df = pd.read_csv(data_path)
print("Data loaded successfully!")

df.head()

# Check data structure
df.info()


# ======= Data Cleaning =======
# Calculate the number of missing values in each column
missing_values = df.isnull().sum()

# Calculate the percentage of missing values
missing_percentage = (missing_values / len(df)) * 100

# Combine the results into a single DataFrame for easy viewing
missing_info = pd.DataFrame(
    {'Missing Values': missing_values, 'Percentage': missing_percentage})

# Display columns with missing values, sorted by percentage
print(missing_info[missing_info['Missing Values'] >
      0].sort_values(by='Percentage', ascending=False))


# Handling Missing Values
df['Engagement_with_Ads'] = df['Engagement_with_Ads'].fillna('None')
df['Social_Media_Influence'] = df['Social_Media_Influence'].fillna('None')
print(df.isnull().sum())


# Correct Data Types
df['Purchase_Amount'] = df['Purchase_Amount'].str.replace(
    '$', '', regex=False).astype(float)
print("Data type of 'Purchase_Amount' after cleaning:")
print(df['Purchase_Amount'].dtype)

# Correct Data Types
df['Time_of_Purchase'] = pd.to_datetime(
    df['Time_of_Purchase'], errors='coerce')
print("Data type of 'Time_of_Purchase' after cleaning:")
print(df['Time_of_Purchase'].dtype)


# ======= Data Exploration =======
# Summary statistics
pd.set_option('display.max_columns', None)
df.describe(include='all')

# Distribution of key categorical variables
print("Purchase Category Distribution:\n",
      df['Purchase_Category'].value_counts().head(10))

# Distribution of key numerical variables

sns.set_style('whitegrid')

plt.figure(figsize=(10, 6))
sns.histplot(df['Customer_Satisfaction'], bins=30, kde=True)
plt.title('Distribution of Customer_Satisfaction')
plt.xlabel('Customer_Satisfaction')
plt.ylabel('Frequency')
save_plot(plt, "ecommerce_customer_satisfaction_distribution.png")
plt.show()


#  ======= RMF Customer Segmentation =======

# Calculate R, F, M Values
snapshot_date = df['Time_of_Purchase'].max() + dt.timedelta(days=1)
print(f"Snapshot Date: {snapshot_date}")

rfm_df = df.groupby('Customer_ID').agg({
    # Recency: Days since last purchase
    'Time_of_Purchase': lambda date: (snapshot_date - date.max()).days,
    # Frequency: Total number of purchases
    'Frequency_of_Purchase': 'sum',
    # Monetary: Total amount spent
    'Purchase_Amount': 'sum'
})

rfm_df.rename(columns={'Time_of_Purchase': 'Recency',
                       'Frequency_of_Purchase': 'Frequency',
                       'Purchase_Amount': 'Monetary'}, inplace=True)

print("\nRFM DataFrame:")
print(rfm_df.head())

# Create R, F, M Scores & is_churn
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(
    method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm_df['is_churn'] = (rfm_df['Recency'] > 180).astype(int)

print("\nRFM DataFrame with Scores:")
print(rfm_df.head())

# Customer Segmentation
rfm_df['RF_Segment'] = rfm_df['R_Score'].astype(
    str) + rfm_df['F_Score'].astype(str)
# rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

segment_map = {
    # Not purchased recently & rarely buy - Dormant/Lost customers
    r'[1-2][1-2]': 'Hibernating',
    # Not recent but previously active - Risk of churn
    r'[1-2][3-4]': 'At-Risk',
    # Not recent but were highly active - High-value at risk
    r'[1-2]5': 'Cannot Lose Them',
    # Moderately recent but low frequency - Becoming inactive
    r'3[1-2]': 'About to Sleep',
    # Average recency & frequency - Need re-engagement
    r'33': 'Need Attention',
    # Fairly recent & frequent purchases - Core loyal base
    r'[3-4][4-5]': 'Loyal Customers',
    # Recent but only one purchase - Promising newcomers
    r'41': 'Promising',
    # Very recent first purchase - Brand new customers
    r'51': 'New Customers',
    # Recent with some repetition - Developing loyalty
    r'[4-5][2-3]': 'Potential Loyalists',
    # Very recent & very frequent - Best customers
    r'5[4-5]': 'Champions'
}

rfm_df['Segment'] = rfm_df['RF_Segment'].replace(segment_map, regex=True)

print("\nRFM DataFrame with Segment:")
print(rfm_df.head())


# ======= Feature Engineering =======
# Merge the RFM clustering results back into the main table
customer_segments = rfm_df.reset_index(
)[['Customer_ID', 'Segment', 'is_churn', 'Recency', 'Frequency', 'Monetary']]
df = pd.merge(df, customer_segments, on='Customer_ID', how='left')

# Data preparation
y = df['is_churn']

features_df = df.drop(
    columns=['Customer_ID', 'Time_of_Purchase', 'Location', 'is_churn', 'Recency', 'Frequency'])

features_df['Customer_Loyalty_Program_Member'] = features_df['Customer_Loyalty_Program_Member'].astype(
    int)
features_df['Discount_Used'] = features_df['Discount_Used'].astype(int)

numerical_cols = features_df.select_dtypes(
    include=['int64', 'float64']).columns
categorical_cols = features_df.select_dtypes(include=['object']).columns

print("Numerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# One-Hot Encoding
encoded_df = pd.get_dummies(
    features_df, columns=categorical_cols, drop_first=True, dtype=int)

# Normalization

scaler = StandardScaler()
encoded_df[numerical_cols] = scaler.fit_transform(encoded_df[numerical_cols])

X = encoded_df
X.head()


# ======= Machine Learning with LR =======

# --- Split the data into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("--- Data Split ---")
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target distribution:", pd.Series(
    y_train).value_counts(normalize=True))
print("Testing target distribution:", pd.Series(
    y_test).value_counts(normalize=True))

# --- Initialize and train the Logistic Regression model ---
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)  # Train the model on training data

# --- Make predictions and evaluate model performance ---
y_pred = model.predict(X_test)  # Predict labels for test set

print("\n--- Model Performance ---")
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Accuracy: {accuracy:.4f}")

# Get coefficients to understand which features drive predictions
if hasattr(model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))


# ======= Machine Learning with XGBoost =======

# --- Initialize and train the XGBoost model ---
xgb_model = XGBClassifier(
    n_estimators=100,        # Number of trees (boosting rounds)
    random_state=42,         # For reproducible results
    learning_rate=0.1,       # Step size shrinkage to prevent overfitting
    max_depth=3,             # Maximum tree depth
    subsample=0.8,           # Subsample ratio of the training instances
    colsample_bytree=0.8,    # Subsample ratio of columns when constructing each tree
    eval_metric='logloss',   # Evaluation metric for binary classification
    use_label_encoder=False  # Avoid warning messages
)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)

# Calculate accuracy and AUC
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba[:, 1])

print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"XGBoost AUC: {xgb_auc:.4f}")

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, xgb_pred_proba[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(
    fpr, tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})', color='darkgreen', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc='lower right')
plt.grid(True)
save_plot(plt, "ecommerce_xgboost_roc_curve.png")
plt.show()


# ======= Visualization =======
# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'].head(15),
         feature_importance['importance'].head(15))
plt.title('Top 15 Feature Importance - XGBoost')
plt.xlabel('Importance Score')
plt.tight_layout()
save_plot(plt, "ecommerce_xgboost_feature_importance.png")
plt.show()
