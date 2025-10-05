from src.features import add_high_satisfaction_label, select_and_encode_features


def test_add_high_satisfaction_label(tiny_df):
    y = add_high_satisfaction_label(tiny_df, threshold=7)

    assert y.tolist() == [0, 1, 1, 0]
    assert y.name == "High_Satisfaction"


def test_shape_and_no_nulls(tiny_df):
    X, numeric_cols = select_and_encode_features(tiny_df)
    assert X.shape[0] == len(tiny_df)
    assert X.isnull().sum().sum() == 0

    for col in numeric_cols:
        assert col in X.columns
