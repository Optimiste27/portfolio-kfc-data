# src/check_product_mapping.py
import pandas as pd
import sqlite3

conn = sqlite3.connect('cos_kfc.db')

# Lire les transactions CSV pour voir tous les produits uniques
df = pd.read_csv('../data/raw/transactions_final_20260124_1112.csv', encoding='utf-8')
all_products = df['product_family'].unique()
print("Produits uniques dans le CSV:")
for product in sorted(all_products):
    print(f"  - {product}")

# Voir les familles dans la base
cursor = conn.cursor()
cursor.execute('SELECT product_family_name FROM product_families ORDER BY product_family_name')
db_products = [row[0] for row in cursor.fetchall()]
print("\nProduits dans la base de données:")
for product in db_products:
    print(f"  - {product}")

# Vérifier les transactions problématiques
cursor.execute('''
    SELECT DISTINCT t.product_family_id, p.product_family_name
    FROM transactions t
    LEFT JOIN product_families p ON t.product_family_id = p.product_family_id
    WHERE p.product_family_id IS NULL
    LIMIT 10
''')
print("\nTransactions avec ID produit invalide (exemples):")
for row in cursor.fetchall():
    print(f"  product_family_id: {row[0]}")

conn.close()