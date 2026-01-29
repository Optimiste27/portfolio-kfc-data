# Créez ce script d'analyse : src/analyze_data_structure.py
import pandas as pd
import sqlite3

# Analyser les CSV
restaurants_df = pd.read_csv('../data/raw/restaurants_final_20260124_1112.csv')
transactions_df = pd.read_csv('../data/raw/transactions_final_20260124_1112.csv')

print("=== ANALYSE RESTAURANTS ===")
print(f"Lignes: {len(restaurants_df)}")
print(f"Colonnes: {list(restaurants_df.columns)}")
print("\nValeurs uniques performance_category:")
print(restaurants_df['performance_category'].value_counts())

print("\n=== ANALYSE TRANSACTIONS ===")
print(f"Lignes: {len(transactions_df)}")
print(f"Produits uniques: {transactions_df['product_family'].nunique()}")
print("Produits:")
print(transactions_df['product_family'].value_counts())