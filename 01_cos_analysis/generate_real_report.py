"""
Génère un rapport PDF avec les données réelles du projet
"""

import pandas as pd
import sys
import os
from datetime import datetime

# Ajoute le chemin src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("📊 GÉNÉRATION DE RAPPORT AVEC DONNÉES RÉELLES")
    print("=" * 50)
    
    try:
        from pdf_reporter import COSPDFReporter
        print("✅ Module PDF disponible")
    except ImportError as e:
        print(f"❌ Module PDF non disponible: {e}")
        print("\n💡 Solution: pip install reportlab")
        return False
    
    # Chemin des données
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    
    # Cherche les fichiers les plus récents
    import glob
    
    # Transactions
    trans_files = glob.glob(os.path.join(data_dir, "transactions*.csv"))
    if not trans_files:
        print(f"❌ Aucun fichier de transactions trouvé dans {data_dir}")
        return False
    
    latest_trans = max(trans_files, key=os.path.getctime)
    print(f"📁 Fichier transactions: {os.path.basename(latest_trans)}")
    
    # Restaurants
    rest_files = glob.glob(os.path.join(data_dir, "restaurant*.csv"))
    if rest_files:
        latest_rest = max(rest_files, key=os.path.getctime)
        print(f"📁 Fichier restaurants: {os.path.basename(latest_rest)}")
    else:
        print("⚠️  Aucun fichier restaurants trouvé")
        latest_rest = None
    
    # Chargement des données
    print("\n📥 Chargement des données...")
    transactions = pd.read_csv(latest_trans)
    
    if latest_rest:
        restaurants = pd.read_csv(latest_rest)
    else:
        restaurants = pd.DataFrame()
    
    # Aperçu des données
    print(f"✅ Transactions: {len(transactions):,} lignes")
    print(f"✅ Colonnes: {list(transactions.columns)}")
    print(f"📅 Période: {transactions['date'].min()} à {transactions['date'].max()}")
    print(f"📊 Écart moyen: {transactions['gap_percentage'].mean():.2f}%")
    print(f"🏪 Restaurants: {transactions['restaurant_name'].nunique()}")
    print(f"🍗 Produits: {transactions['product_family'].nunique()}")
    
    # Crée le générateur PDF
    print("\n🔄 Création du générateur PDF...")
    output_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    reporter = COSPDFReporter(output_dir=output_dir)
    
    # Génère différents rapports
    print("\n📄 Génération des rapports...")
    
    # 1. Rapport hebdomadaire
    print("1. Rapport hebdomadaire (semaine 1)...")
    weekly_report = reporter.generate_weekly_report(
        transactions, 
        restaurants,
        week_number=1
    )
    
    # 2. Rapports par restaurant (top 3 avec plus d'écarts)
    print("\n2. Rapports par restaurant...")
    restaurant_perf = transactions.groupby('restaurant_name')['gap_percentage'].mean()
    top_3_problematic = restaurant_perf.sort_values(ascending=False).head(3)
    
    restaurant_reports = []
    for restaurant, gap in top_3_problematic.items():
        print(f"   • {restaurant} ({gap:.1f}%)...")
        try:
            report = reporter.generate_restaurant_specific_report(transactions, restaurant)
            if report:
                restaurant_reports.append(report)
                print(f"      ✅ Généré")
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
    
    # 3. Rapport mensuel
    print("\n3. Rapport mensuel (janvier 2024)...")
    try:
        monthly_report = reporter.generate_monthly_report(
            transactions,
            restaurants,
            month=1,
            year=2024
        )
        if monthly_report:
            print(f"   ✅ Généré")
        else:
            print(f"   ⚠️  Aucune donnée")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Résumé
    print("\n" + "=" * 50)
    print("📁 RÉSULTATS")
    print(f"Dossier des rapports: {os.path.abspath(output_dir)}")
    
    # Liste tous les fichiers PDF
    pdf_files = glob.glob(os.path.join(output_dir, "*.pdf"))
    
    if pdf_files:
        print(f"\n📄 RAPPORTS GÉNÉRÉS ({len(pdf_files)} fichiers):")
        
        # Trie par date de modification
        pdf_files.sort(key=os.path.getmtime, reverse=True)
        
        for i, pdf_file in enumerate(pdf_files[:5], 1):  # 5 derniers
            filename = os.path.basename(pdf_file)
            size = os.path.getsize(pdf_file)
            modified = datetime.fromtimestamp(os.path.getmtime(pdf_file))
            
            print(f"\n{i}. {filename}")
            print(f"   📏 Taille: {size:,} octets")
            print(f"   🕐 Généré: {modified.strftime('%d/%m/%Y %H:%M')}")
            
            # Aperçu du contenu pour les plus gros fichiers
            if size > 1000:
                print(f"   📊 Contenu: Rapport complet")
    
    else:
        print("\n⚠️  Aucun rapport généré")
    
    print("\n✅ GÉNÉRATION TERMINÉE")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)