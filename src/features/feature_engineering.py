import pandas as pd
import numpy as np
from datetime import timedelta
from src.logger import logging
import os


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level features for CLV modeling using
    a rolling 90-day cutoff window.
    """

    try:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        cutoff_date = df["InvoiceDate"].max() - timedelta(days=90)

        logging.info("Using cutoff date: %s", cutoff_date.date())

        # Feature window
        df_features = df[df["InvoiceDate"] <= cutoff_date].copy()
        df_clv = df[df['InvoiceDate'] > cutoff_date]

        # Aggregate customer-level features
        customer_features = df_features.groupby("Customer ID").agg(
            first_purchase_date=("InvoiceDate", "min"),
            last_purchase_date=("InvoiceDate", "max"),
            unique_invoices=("Invoice", "nunique"),
            total_quantity=("Quantity", "sum"),
            avg_quantity_per_order=("Quantity", "mean"),
            unit_price_std=("Price", "std"),
        ).round(2)

        # Time-based features
        customer_features["customer_age_days"] = (
            cutoff_date - customer_features["first_purchase_date"]
        ).dt.days

        customer_features['days_since_last_purchase'] = (cutoff_date - customer_features['last_purchase_date']).dt.days


        # Behavioral ratios

        customer_features['average_days_between_purchase'] = customer_features['customer_age_days'] / customer_features['unique_invoices']

        customer_features['is_onetime_buyer'] = (customer_features['unique_invoices']==1).astype(int)


        # Handle NaNs
        customer_features["unit_price_std"] = (
            customer_features["unit_price_std"].fillna(0)
        )

        #caluclate target clv
        clv_data = df_clv.groupby('Customer ID')['Total Amount'].sum().reset_index()
        clv_data.columns = ['Customer ID', 'target_clv']

        #Merge caluclated clv to customer features
        customer_data = customer_features.reset_index().merge(clv_data,on='Customer ID',how='inner')

        #Log transform target clv
        customer_data['target_clv'] = np.log1p(customer_data['target_clv'])

        # DROP FEATURES NOT USED IN FINAL MODEL

        customer_data = customer_data.drop(
            columns=[
                "first_purchase_date",
                "last_purchase_date",
                "Customer ID"
            ]
        )

        logging.info(
            "Feature engineering completed. Shape: %s | Columns: %s",
            customer_data.shape,
            customer_data.columns
        )

        return customer_data.reset_index()

    except Exception as e:
        logging.error("Feature engineering failed: %s", e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:


        train_data = load_data('./data/interim/train_data.csv')
        test_data = load_data('./data/interim/test_data.csv')

        train_df  = build_features(train_data)
        test_df  = build_features(test_data)

        save_data(train_df, os.path.join("./data", "processed", "train_data.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_data.csv"))
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
