import pytest
from src.data_utils import clean_currency_column, filter_data, group_data


def test_clean_currency_column_converts_to_float(tiny_df):
    out = clean_currency_column(tiny_df, "Purchase_Amount")
    assert out["Purchase_Amount"].dtype == float
    assert abs(out["Purchase_Amount"].iloc[0] - 100.0) < 1e-6


def test_clean_currency_column_missing_raises(tiny_df):
    with pytest.raises(KeyError):
        clean_currency_column(tiny_df, "does_not_exist")


def test_filter_data_simple_equality(tiny_df):
    out = filter_data(tiny_df, {"Gender": "Female"})
    assert set(out["Customer_ID"]) == {"A", "C", "D"}


def test_group_data_mean(tiny_df):
    out = group_data(
        tiny_df, group_col="Gender", agg_col="Time_to_Decision", agg="mean"
    )
    assert set(out["Gender"]) == {"Female", "Male"}
    male = out.loc[out["Gender"] == "Male", "Time_to_Decision"].iloc[0]
    assert male == 6.0
