# ml/check_dataset.py
"""
Vérification du dataset créé
"""

import pandas as pd
import numpy as np

print("🔍 VÉRIFICATION DU DATASET ML")
print("="*60)

# Charger le dataset créé
dataset_path = '../data/processed/ml_dataset.csv'
df = pd.read_csv(dataset_path)

print(f"📊 Dataset: {len(df)} lignes, {len(df.columns)} colonnes")
print(f"\n📐 Types de données:")
print(df.dtypes.value_counts())

print(f"\n🎯 Variable cible (gap_percentage):")
print(f"  • Moyenne: {df['gap_percentage'].mean():.2f}%")
print(f"  • Std: {df['gap_percentage'].std():.2f}%")
print(f"  • Min: {df['gap_percentage'].min():.2f}%")
print(f"  • Max: {df['gap_percentage'].max():.2f}%")

print(f"\n🔍 Vérification des NaN:")
nan_counts = df.isnull().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) > 0:
    print("Colonnes avec NaN:")
    for col, count in nan_cols.items():
        print(f"  • {col}: {count} NaN")
else:
    print("✅ Aucun NaN détecté")

print(f"\n📈 Statistiques des features numériques (top 10):")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Nombre de features numériques: {len(numeric_cols)}")

# Vérifier les corrélations (uniquement avec les colonnes numériques)
if 'gap_percentage' in df.columns:
    correlations = df[numeric_cols].corr()['gap_percentage'].abs().sort_values(ascending=False)
    print(f"\n🔗 Top 10 features corrélées avec gap_percentage:")
    for i, (feat, corr) in enumerate(correlations[1:11].items(), 1):
        print(f"  {i:2}. {feat:30}: {corr:.3f}")

print("\n✅ Dataset prêt pour l'entraînement ML!")