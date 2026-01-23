import pandas as pd
import numpy as np

print("🔍 EXPLORATION DES DONNÉES COS GÉNÉRÉES")
print("=" * 60)

# Chargement des données
transactions = pd.read_csv('data/raw/transactions_daily.csv')
restaurants = pd.read_csv('data/raw/restaurant_info.csv')

print(f"📊 TRANSACTIONS:")
print(f"• Nombre total: {len(transactions):,}")
print(f"• Période: {transactions['date'].min()} à {transactions['date'].max()}")
print(f"• Colonnes: {list(transactions.columns)}")

print(f"\n🏪 RESTAURANTS:")
print(f"• Nombre: {len(restaurants)}")
print(f"• Villes: {restaurants['city'].unique()}")

print(f"\n📈 KPIS CLÉS:")
print(f"• CA total: {transactions['revenue'].sum():,.0f} €")
print(f"• Coût total réel: {(transactions['actual_unit_cost'] * transactions['units_sold']).sum():,.0f} €")
print(f"• Gaspillage total: {transactions['waste_kg'].sum():,.1f} kg")
print(f"• Coût gaspillage: {transactions['waste_cost'].sum():,.0f} €")

print(f"\n🔍 DISTRIBUTION PAR PRODUIT:")
product_stats = transactions.groupby('product_family').agg({
    'gap_percentage': 'mean',
    'waste_kg': 'mean',
    'revenue': 'sum',
    'units_sold': 'sum'
}).round(2)

print(product_stats.sort_values('gap_percentage', ascending=False))

print(f"\n📅 TENDANCE TEMPORELLE (premier mois):")
transactions['date'] = pd.to_datetime(transactions['date'])
jan_data = transactions[transactions['date'].dt.month == 1]
daily_gap = jan_data.groupby('date')['gap_percentage'].mean()

print(f"Écart moyen janvier: {daily_gap.mean():.2f}%")
print(f"Jour max écart: {daily_gap.idxmax().date()} ({daily_gap.max():.2f}%)")

print(f"\n🚨 ANALYSE ANOMALIES:")
if 'anomaly_type' in transactions.columns:
    anomalies = transactions[transactions['operational_anomaly'] == True]
    if not anomalies.empty:
        print(f"Total anomalies: {len(anomalies):,}")
        print("Types d'anomalies:")
        print(anomalies['anomaly_type'].value_counts())
        
        print(f"\nImpact des anomalies sur l'écart:")
        anomaly_impact = anomalies.groupby('anomaly_type')['gap_percentage'].mean().round(2)
        print(anomaly_impact.sort_values(ascending=False))

print(f"\n✅ Pour explorer plus:")
print("1. transactions.head(10) - Voir les premières lignes")
print("2. transactions.describe() - Statistiques descriptives")
print("3. transactions['restaurant_name'].value_counts() - Distribution restaurants")

# Sauvegarde un résumé
summary_stats = transactions.groupby(['restaurant_name', 'product_family']).agg({
    'gap_percentage': 'mean',
    'revenue': 'sum',
    'waste_kg': 'sum'
}).round(2)

summary_stats.to_csv('data/processed/summary_by_restaurant_product.csv')
print(f"\n💾 Résumé sauvegardé: data/processed/summary_by_restaurant_product.csv")