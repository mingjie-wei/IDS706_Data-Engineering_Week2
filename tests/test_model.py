import pytest
import pandas as pd
from src.features import add_high_satisfaction_label, select_and_encode_features
from src.data_utils import clean_currency_column
from src.model import train_logreg_classifier


def test_train_logreg_classifier_end_to_end(tiny_df):
    df = clean_currency_column(tiny_df, 'Purchase_Amount')
    X, _ = select_and_encode_features(df)
    y = add_high_satisfaction_label(df)
    metrics = train_logreg_classifier(X, y, test_size=0.5, random_state=0)
    assert 'model' in metrics and metrics['n_train'] > 0 and metrics['n_test'] > 0
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['roc_auc'] <= 1.0


def test_train_logreg_raises_on_single_class(tiny_df):
    df = tiny_df.copy()
    df['Customer_Satisfaction'] = 10  # single class
    X, _ = select_and_encode_features(df)
    y = add_high_satisfaction_label(df)
    with pytest.raises(ValueError):
        train_logreg_classifier(X, y)
