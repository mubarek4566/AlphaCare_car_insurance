import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class visualize:
    def __init__(self, path):
        self.df = path

    def plot_numeric_distributions(self, num_cols, bins=30):
        """
        Plots histograms for a list of numerical columns.
        """
        for col in num_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col].dropna(), bins=bins, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

    def plot_categorical_distributions(self, cat_cols, top_n=10):
        """
        Plots bar charts for a list of categorical columns.
        """
        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            value_counts = self.df[col].value_counts().nlargest(top_n)
            sns.barplot(x=value_counts.values, y=value_counts.index)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()


    def aggregate_by_month_zip(self):
        """
        Aggregates TotalPremium and TotalClaims by TransactionMonth and PostalCode.
        """
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
        monthly_zip_summary = (
            self.df.groupby([self.df['TransactionMonth'].dt.to_period("M").astype(str), 'PostalCode'])
            [['TotalPremium', 'TotalClaims']]
            .sum()
            .reset_index()
            .rename(columns={"TransactionMonth": "Month"})
        )
        return monthly_zip_summary

    def plot_premium_vs_claims_scatter(self, monthly_zip_summary):
        """
        Plots scatter plot of TotalPremium vs TotalClaims by PostalCode over time.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=monthly_zip_summary,
            x="TotalPremium", 
            y="TotalClaims", 
            hue="PostalCode", 
            alpha=0.7, 
            legend=False
        )
        plt.title("Scatter Plot: TotalPremium vs TotalClaims by PostalCode (Monthly Aggregated)")
        plt.xlabel("Total Premium")
        plt.ylabel("Total Claims")
        plt.tight_layout()
        plt.show()

    def correlation_matrix_plot(self, numeric_cols):
        """
        Plots correlation matrix for selected numeric columns.
        """
        corr = self.df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix of Numeric Variables")
        plt.tight_layout()
        plt.show()


    # =========================================================

    def average_premium_by_province(self):
        """
        Plots average TotalPremium per Province.
        """
        avg_premium = self.df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=avg_premium.values, y=avg_premium.index, palette='Blues_r')
        plt.title("Average Total Premium by Province")
        plt.xlabel("Average Premium")
        plt.ylabel("Province")
        plt.tight_layout()
        plt.show()

    def cover_type_distribution_by_province(self, top_n=10):
        """
        Plots distribution of CoverType across top Provinces.
        """
        top_provinces = self.df['Province'].value_counts().nlargest(top_n).index
        filtered = self.df[self.df['Province'].isin(top_provinces)]
        plt.figure(figsize=(12, 6))
        sns.countplot(data=filtered, x='Province', hue='CoverType')
        plt.title("Distribution of Cover Types by Province")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(title='Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    def top_vehicle_makes_by_province(self, province, top_n=10):
        """
        Plots top vehicle makes in a specific Province.
        """
        province_df = self.df[self.df['Province'] == province]
        top_makes = province_df['make'].value_counts().nlargest(top_n)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=top_makes.values, y=top_makes.index, palette='Set2')
        plt.title(f"Top {top_n} Vehicle Makes in {province}")
        plt.xlabel("Count")
        plt.ylabel("Vehicle Make")
        plt.tight_layout()
        plt.show()

    # =========================================================

    def trend_avg_premium_by_province(self, top_n=5):
        """
        Plots trend of average TotalPremium by Province over time.
        """
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
        top_provinces = self.df['Province'].value_counts().nlargest(top_n).index
        filtered = self.df[self.df['Province'].isin(top_provinces)]

        trend_data = (
            filtered.groupby([self.df['TransactionMonth'].dt.to_period("M").astype(str), 'Province'])['TotalPremium']
            .mean()
            .reset_index()
            .rename(columns={"TransactionMonth": "Month", "TotalPremium": "AvgPremium"})
        )

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=trend_data, x="Month", y="AvgPremium", hue="Province", marker="o")
        plt.title("Trend of Average Premium by Province Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def trend_cover_type_distribution(self, top_n=3):
        """
        Plots the trend in count of CoverType usage over time in top Provinces.
        """
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
        top_provinces = self.df['Province'].value_counts().nlargest(top_n).index
        filtered = self.df[self.df['Province'].isin(top_provinces)]

        trend = (
            filtered.groupby([self.df['TransactionMonth'].dt.to_period("M").astype(str), 'Province', 'CoverType'])
            .size()
            .reset_index(name='Count')
            .rename(columns={'TransactionMonth': 'Month'})
        )

        g = sns.FacetGrid(trend, col="Province", hue="CoverType", col_wrap=top_n, height=4, sharey=False)
        g.map(sns.lineplot, "Month", "Count", marker="o")
        g.add_legend()
        g.set_xticklabels(rotation=45)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Trend of Cover Types Over Time by Province")
        plt.tight_layout()
        plt.show()

    def trend_top_vehicle_makes(self, province, top_n=5):
        """
        Plots time trends of top vehicle makes in a specific province.
        """
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
        filtered = self.df[self.df['Province'] == province]
        top_makes = filtered['make'].value_counts().nlargest(top_n).index
        filtered = filtered[filtered['make'].isin(top_makes)]

        trend = (
            filtered.groupby([self.df['TransactionMonth'].dt.to_period("M").astype(str), 'make'])
            .size()
            .reset_index(name='Count')
            .rename(columns={'TransactionMonth': 'Month'})
        )

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=trend, x='Month', y='Count', hue='make', marker="o")
        plt.title(f"Trend of Top {top_n} Vehicle Makes Over Time in {province}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def detect_outliers_boxplot(self, numeric_cols=None, top_n=6):
        """
        Uses box plots to visualize outliers in specified numerical columns.
        """
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]

        num_cols = len(numeric_cols)
        num_plots = min(top_n, num_cols)
        nrows = (num_cols + num_plots - 1) // num_plots

        for i in range(nrows):
            cols_to_plot = numeric_cols[i * top_n: (i + 1) * top_n]
            if not cols_to_plot:
                break

            plt.figure(figsize=(4 * len(cols_to_plot), 5))
            for idx, col in enumerate(cols_to_plot, 1):
                plt.subplot(1, len(cols_to_plot), idx)
                sns.boxplot(y=self.df[col], color='skyblue')
                plt.title(f'Boxplot of {col}')
                plt.tight_layout()

            plt.show()
