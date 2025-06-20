# import required libraries due to state reset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
import joblib
from datetime import datetime

class DataPreparation:
    def __init__(self):
        self.df = {}
        
    # Define functions for data preparation
    def handle_missing_data(self, df):
        """
        Imputes missing numerical and categorical values.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include='object').columns

        # Impute numeric columns with median
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

        # Impute categorical columns with most frequent
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

        return df

    def feature_engineering(self, df):
        # Age of Vehicle (difference between transaction date and vehicle intro date)
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'])
        # Age of Vehicle: Based on VehicleIntroDate and TransactionMonth
        df['VehicleAge'] = (df['TransactionMonth'] - df['VehicleIntroDate']).dt.days / 365
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['year'] = df['TransactionMonth'].dt.year
        df['month'] = df['TransactionMonth'].dt.month
        df['day'] = df['TransactionMonth'].dt.day
        # Claim Frequency: Total claims over a specific period divided by the number of transactions.
        df['ClaimFrequency'] = df['TotalClaims'] / df['TransactionMonth'].dt.month
        df = df.drop(columns=['TransactionMonth', 'VehicleIntroDate'])  # Drop original date column if not needed
        return df
    def categorical_encoding(self, df):
        # One-Hot Encoding for features with more than two unique categories
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Label Encoding for binary features
        le = LabelEncoder()
        bool_cols = df.select_dtypes(include = 'bool').columns
        for cols in bool_cols:
            df[cols] = le.fit_transform(df[cols])
        
        return df

    def Train_Test_Split(self, df, test_size):
        # Define your target variable and features
        X = df.drop(columns=['TotalPremium', 'TotalClaims'])
        y = df['TotalClaims']  # or 'TotalPremium' based on your target variable

        # Split into 80% train and 20% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test 

    