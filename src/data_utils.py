import pandas as pd


def load_csv(file_path):
    # Load CSV file into DataFrame
    df = pd.read_csv(file_path, encoding='utf-8',
                     na_values=['', 'NA', 'N/A', 'null'])

    return df


def clean_currency_column(df, column_name):
    # Clean currency column by removing '$', ',', and converting to float
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")

    result = df.copy()

    # Steps to clean currency format:
    # 1. Convert to string
    # 2. Remove dollar sign ($)
    # 3. Remove commas (,)
    # 4. Remove leading/trailing whitespace
    # 5. Convert to float
    result[column_name] = result[column_name].astype(str)
    result[column_name] = result[column_name].str.replace('$', '')
    result[column_name] = result[column_name].str.replace(',', '')
    result[column_name] = result[column_name].str.strip()
    result[column_name] = result[column_name].astype(float)

    return result


def filter_data(df, filter_dict):
    # Filter DataFrame based on conditions in filter_dict
    result = df.copy()

    for column, value in filter_dict.items():
        if column not in result.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        result = result[result[column] == value]

    return result


def group_data(df, *, group_col, agg_col, agg="mean"):
    # Group data and perform aggregation calculations
    missing = [c for c in (group_col, agg_col) if c not in df.columns]

    if missing:
        raise KeyError(f"Missing column(s): {missing}")

    if df.empty:
        return pd.DataFrame(columns=[group_col, agg_col])

    grouped = (
        df.groupby(group_col, dropna=False)[agg_col]
        .agg(agg)
        .reset_index()
    )

    return grouped
