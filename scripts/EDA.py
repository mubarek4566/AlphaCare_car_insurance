# Re-import necessary libraries after kernel reset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, path):
        self.df = path

    # Define reusable functions for analysis
    def compute_loss_ratio_stats(self):
        # Convert relevant columns to numeric
        self.df["TotalPremium"] = pd.to_numeric(self.df["TotalPremium"], errors="coerce")
        self.df["TotalClaims"] = pd.to_numeric(self.df["TotalClaims"], errors="coerce")
        self.df = self.df[self.df["TotalPremium"] > 0]

        # Compute overall loss ratio
        overall_loss_ratio = self.df["TotalClaims"].sum() / self.df["TotalPremium"].sum()

        # Group loss ratios
        by_province = self.df.groupby("Province")[["TotalClaims", "TotalPremium"]].sum()
        by_province["LossRatio"] = by_province["TotalClaims"] / by_province["TotalPremium"]

        by_vehicle_type = self.df.groupby("VehicleType")[["TotalClaims", "TotalPremium"]].sum()
        by_vehicle_type["LossRatio"] = by_vehicle_type["TotalClaims"] / by_vehicle_type["TotalPremium"]

        by_gender = self.df.groupby("Gender")[["TotalClaims", "TotalPremium"]].sum()
        by_gender["LossRatio"] = by_gender["TotalClaims"] / by_gender["TotalPremium"]

        return overall_loss_ratio, by_province, by_vehicle_type, by_gender

    def plot_loss_ratios(self, group_col):
        plt.figure(figsize=(10, 6))
        group = self.df.groupby(group_col)[["TotalClaims", "TotalPremium"]].sum()
        group["LossRatio"] = group["TotalClaims"] / group["TotalPremium"]
        group = group.sort_values("LossRatio", ascending=False)
        sns.barplot(data=group, x=group_col, y="LossRatio", hue=group_col, palette="coolwarm", legend=False)
        plt.title(f"Loss Ratio by {group_col}")
        plt.xticks(rotation=45)
        plt.ylabel("Loss Ratio")
        plt.tight_layout()
        plt.show()

    def outliers_distribution(self, columns):
        stats = {}
        for col in columns:
            desc = self.df[col].describe()
            q1 = desc["25%"]
            q3 = desc["75%"]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            stats[col] = {
                "summary": desc,
                "outlier_count": len(outliers),
                "outlier_percentage": len(outliers) / len(self.df) * 100,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound
            }

            # Plot boxplot
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

        return stats

    def preprocess_transaction_month(self, date_column='TransactionMonth'):
        """
        Convert TransactionMonth column to datetime format.
        """
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        return self.df

    def compute_monthly_claim_summary(self, date_column='TransactionMonth', claims_column='TotalClaims'):
        """
        Group data by month and compute claim frequency, total claims, and claim severity.
        """
        monthly_summary = self.df.groupby(self.df[date_column].dt.to_period("M")).agg(
            ClaimFrequency=(claims_column, lambda x: (x > 0).sum()),
            TotalClaims=(claims_column, 'sum')
        )
        monthly_summary['ClaimSeverity'] = monthly_summary['TotalClaims'] / monthly_summary['ClaimFrequency']
        return monthly_summary.fillna(0)

    def plot_claim_trends(self, monthly_summary):
        """
        Plot claim frequency and severity over time.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot Claim Frequency
        monthly_summary['ClaimFrequency'].plot(ax=axes[0], marker='o', color='blue')
        axes[0].set_title("Claim Frequency Over Time")
        axes[0].set_ylabel("Number of Claims")

        # Plot Claim Severity
        monthly_summary['ClaimSeverity'].plot(ax=axes[1], marker='o', color='green')
        axes[1].set_title("Claim Severity Over Time")
        axes[1].set_ylabel("Average Claim Amount")
        axes[1].set_xlabel("Transaction Month")

        plt.tight_layout()
        plt.show()
