"""
Test de génération de rapports PDF - VERSION CORRIGÉE
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdf_reporter import COSPDFReporter
import pandas as pd
import numpy as np
from datetime import datetime

def create_test_data(n_rows=500):
    """Crée des données de test réalistes avec toutes les colonnes nécessaires"""
    print(f"Création de {n_rows} lignes de données de test...")
    
    np.random.seed(42)
    
    # Génère les dates de manière cohérente
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    # Crée les données colonne par colonne
    data = {
        'transaction_id': range(10000, 10000 + n_rows),
        'date': np.random.choice(dates, n_rows),
        'restaurant_id': np.random.choice([1, 2, 3, 4, 5, 6], n_rows),
        'restaurant_name': np.random.choice(
            ['KFC Paris Louvre', 'KFC Lyon Part-Dieu', 'KFC Marseille Vieux Port',
             'KFC Toulouse Capitole', 'KFC Lille Centre', 'KFC Bordeaux Quinconces'],
            n_rows
        ),
        'product_family': np.random.choice(
            ['Poulet Frit', 'Sandwichs', 'Burger Poulet', 'Wraps & Salades',
             'Accompagnements', 'Boissons', 'Desserts', 'Petits Déjeuners'],
            n_rows
        ),
        'theoretical_unit_cost': np.random.uniform(1.0, 5.0, n_rows),
        'units_sold': np.random.randint(1, 20, n_rows),
        'selling_price': np.random.uniform(2.0, 10.0, n_rows),
        'waste_kg': np.random.uniform(0, 3, n_rows),
        'qsp_score': np.random.uniform(0.7, 0.95, n_rows),
        'operational_anomaly': np.random.random(n_rows) > 0.85,
        'gap_percentage': np.random.normal(4.0, 1.5, n_rows)
    }
    
    # Calcule les colonnes dérivées
    df = pd.DataFrame(data)
    
    # Revenue = selling_price * units_sold
    df['revenue'] = df['selling_price'] * df['units_sold']
    
    # actual_unit_cost basé sur theoretical_unit_cost et gap_percentage
    df['actual_unit_cost'] = df['theoretical_unit_cost'] * (1 + df['gap_percentage'] / 100)
    
    # Calcule les autres colonnes nécessaires
    df['theoretical_cos'] = (df['theoretical_unit_cost'] * df['units_sold']) / df['revenue']
    df['actual_cos'] = (df['actual_unit_cost'] * df['units_sold']) / df['revenue']
    df['cos_gap'] = df['actual_cos'] - df['theoretical_cos']
    
    # gap_severity
    conditions = [
        df['gap_percentage'] > 10,
        df['gap_percentage'] > 5,
        df['gap_percentage'] > 2
    ]
    choices = ["CRITIQUE", "ÉLEVÉ", "MODÉRÉ"]
    df['gap_severity'] = np.select(conditions, choices, default="FAIBLE")
    
    return df

def create_restaurant_test_data():
    """Crée des données de test pour les restaurants"""
    return pd.DataFrame({
        'restaurant_id': [1, 2, 3, 4, 5, 6],
        'restaurant_name': ['KFC Paris Louvre', 'KFC Lyon Part-Dieu', 'KFC Marseille Vieux Port',
                           'KFC Toulouse Capitole', 'KFC Lille Centre', 'KFC Bordeaux Quinconces'],
        'city': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Lille', 'Bordeaux'],
        'performance_category': ['Excellent', 'Bon', 'À améliorer', 'Bon', 'Moyen', 'Excellent'],
        'target_gap_percentage': [2.0, 3.0, 7.0, 3.5, 4.5, 2.5],
        'manager_experience': [36, 24, 12, 30, 18, 42],
        'team_size': [25, 22, 20, 24, 21, 26],
        'last_audit_score': [0.92, 0.85, 0.72, 0.88, 0.80, 0.94]
    })

def main():
    print("🧪 TEST DE GÉNÉRATION PDF - VERSION CORRIGÉE")
    print("=" * 50)
    
    # Crée les données de test
    print("\n1. Préparation des données...")
    df = create_test_data(n_rows=200)  # 200 lignes suffisent pour le test
    restaurant_df = create_restaurant_test_data()
    
    print(f"✅ Données transactions: {len(df)} lignes, {len(df.columns)} colonnes")
    print(f"✅ Données restaurants: {len(restaurant_df)} restaurants")
    
    # Aperçu des données
    print("\n📋 Aperçu des données créées:")
    print(f"   Colonnes: {list(df.columns)}")
    print(f"   Écart moyen: {df['gap_percentage'].mean():.2f}%")
    print(f"   Restaurants uniques: {df['restaurant_name'].nunique()}")
    
    # Teste le générateur
    print("\n2. Initialisation du générateur PDF...")
    reporter = COSPDFReporter(output_dir="test_reports")
    
    print("\n3. Génération rapport hebdomadaire...")
    try:
        weekly_report = reporter.generate_weekly_report(df, restaurant_df, week_number=2)
        weekly_success = weekly_report is not None
    except Exception as e:
        print(f"❌ Erreur lors de la génération du rapport hebdomadaire: {e}")
        weekly_success = False
        weekly_report = None
    
    print("\n4. Génération rapport mensuel...")
    try:
        monthly_report = reporter.generate_monthly_report(df, restaurant_df, month=1, year=2024)
        monthly_success = monthly_report is not None
    except Exception as e:
        print(f"❌ Erreur lors de la génération du rapport mensuel: {e}")
        monthly_success = False
        monthly_report = None
    
    print("\n5. Génération rapport spécifique restaurant...")
    try:
        specific_report = reporter.generate_restaurant_specific_report(df, 'KFC Marseille Vieux Port')
        specific_success = specific_report is not None
    except Exception as e:
        print(f"❌ Erreur lors de la génération du rapport spécifique: {e}")
        specific_success = False
        specific_report = None
    
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS DU TEST:")
    print(f"✅ Rapport hebdomadaire: {'GÉNÉRÉ' if weekly_success else 'ÉCHEC'}")
    print(f"✅ Rapport mensuel: {'GÉNÉRÉ' if monthly_success else 'ÉCHEC'}")
    print(f"✅ Rapport spécifique: {'GÉNÉRÉ' if specific_success else 'ÉCHEC'}")
    
    # Vérifie que les fichiers existent
    reports_dir = "test_reports"
    if os.path.exists(reports_dir):
        print(f"\n📁 CONTENU DU DOSSIER '{reports_dir}':")
        files = os.listdir(reports_dir)
        if files:
            for file in files:
                filepath = os.path.join(reports_dir, file)
                size = os.path.getsize(filepath)
                print(f"   📄 {file} ({size:,} octets)")
        else:
            print("   (vide)")
    
    # Résumé
    total_success = sum([weekly_success, monthly_success, specific_success])
    if total_success == 3:
        print("\n🎉 TOUS LES TESTS PDF ONT RÉUSSI !")
    elif total_success > 0:
        print(f"\n⚠️  {total_success}/3 tests ont réussi")
    else:
        print("\n❌ Aucun test n'a réussi")
    
    return total_success > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)