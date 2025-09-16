import pandas as pd
from pathlib import Path
from src.data_utils import load_csv, clean_currency_column, group_data
from src.features import add_high_satisfaction_label, select_and_encode_features
from src.model import train_logreg_classifier

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Ecommerce_Consumer_Behavior_Analysis_Data.csv"
)


def test_system_pipeline_runs_on_real_dataset():
    df = load_csv(DATA_PATH)
    assert len(df) >= 100

    # Clean currency column
    df = clean_currency_column(df, 'Purchase_Amount')

    # Basic grouping sanity check
    grp = group_data(df, 'Device_Used_for_Shopping',
                     'Time_to_Decision', 'mean')
    assert not grp.empty

    # Features + target
    X, _ = select_and_encode_features(df)
    y = add_high_satisfaction_label(df)

    # Ensure shapes align
    assert X.shape[0] == y.shape[0]

    # Train model (smoke test for behavior)
    metrics = train_logreg_classifier(X, y, test_size=0.2, random_state=42)
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['roc_auc'] <= 1.0
