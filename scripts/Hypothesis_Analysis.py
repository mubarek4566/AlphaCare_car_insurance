import pandas as pd
from scipy.stats import ttest_ind, f_oneway
import numpy as np

class hypothesis:
    def __init__(self, path):
        self.df = path

    def compute_kpis(self):
        self.df = self.df.copy()
        self.df["HasClaim"] = self.df["TotalClaims"].apply(lambda x: 1 if x > 0 else 0)
        self.df["ClaimSeverity"] = self.df.apply(lambda row: row["TotalClaims"] / row["HasClaim"] if row["HasClaim"] > 0 else 0, axis=1)
        self.df["Margin"] = self.df["TotalPremium"] - self.df["TotalClaims"]
        return self.df

    def create_ab_groups(self, feature, group_a_value, group_b_value):
        """
        Segments data into Group A and B based on a binary or selected categorical feature.
        """
        group_a = self.df[self.df[feature] == group_a_value]
        group_b = self.df[self.df[feature] == group_b_value]
        return group_a, group_b

    def compare_kpis(self, group_a, group_b, kpi_col):
        """
        Performs T-test for the selected KPI between the two groups
        """
        a = group_a[kpi_col].dropna()
        b = group_b[kpi_col].dropna()
        stat, p_val = ttest_ind(a, b, equal_var=False)
        return p_val

    def run_ab_test(self, feature, group_a_value, group_b_value, kpi_list=["HasClaim", "ClaimSeverity", "Margin"]):
        self.compute_kpis()
        group_a, group_b = self.create_ab_groups(feature, group_a_value, group_b_value)

        results = {}
        for kpi in kpi_list:
            p_val = self.compare_kpis(group_a, group_b, kpi)
            results[f"{kpi} (p-value)"] = p_val
        
        return results, len(group_a), len(group_b)
