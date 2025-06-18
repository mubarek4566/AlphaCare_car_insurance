# import required libraries due to state reset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreparation:
    def __init__(self, path):
        self.df = path
        
    # Define functions for data preparation
    def handle_missing_data(self):
        """
        Imputes missing numerical and categorical values.
        """
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        categorical_cols = self.df.select_dtypes(include='object').columns

        # Impute numeric columns with median
        num_imputer = SimpleImputer(strategy='median')
        self.df[numeric_cols] = num_imputer.fit_transform(self.df[numeric_cols])

        # Impute categorical columns with most frequent
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.df[categorical_cols] = cat_imputer.fit_transform(self.df[categorical_cols])

        return self.df

    def feature_engineering(self):
        """
        Create new features relevant to TotalPremium and TotalClaims.
        """
        self.df["Margin"] = self.df["TotalPremium"] - self.df["TotalClaims"]
        self.df["ClaimFlag"] = self.df["TotalClaims"].apply(lambda x: 1 if x > 0 else 0)
        if "Term" in self.df.columns:
            self.df["PremiumPerTerm"] = self.df["TotalPremium"] / self.df["Term"]
        return self.df

    def encode_categorical(self):
        """
        One-hot encodes categorical variables.
        """
        categorical_cols = self.df.select_dtypes(include='object').columns
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        return self.df

    def split_data(self, target, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets.
        """
        X = self.df.drop(columns=[target])
        y = self.df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    
