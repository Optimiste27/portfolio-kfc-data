import pandas as pd

df = pd.read_csv('../data/raw/transactions_final_20260124_1112.csv')
print(f"Écart COS moyen: {df['gap_percentage'].mean():.2f}%")
print(f"Distribution sévérité:\n{df['gap_severity'].value_counts(normalize=True).round(3)*100}")
print(f"\nTop 3 produits problématiques:")
print(df.groupby('product_family')['gap_percentage'].mean().round(2).sort_values(ascending=False).head(3))