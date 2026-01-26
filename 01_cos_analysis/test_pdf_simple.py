"""
Test SIMPLE de génération PDF - Version minimaliste
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pdf_reporter import COSPDFReporter
    PDF_AVAILABLE = True
except ImportError:
    print("❌ Module pdf_reporter non trouvé")
    PDF_AVAILABLE = False

import pandas as pd
import numpy as np

def create_minimal_test_data():
    """Crée le minimum de données nécessaires"""
    data = {
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'restaurant_name': ['KFC Paris Louvre'] * 5 + ['KFC Lyon Part-Dieu'] * 5,
        'product_family': ['Poulet Frit', 'Sandwichs', 'Accompagnements', 'Boissons', 'Desserts'] * 2,
        'gap_percentage': [3.5, 4.2, 7.8, 2.1, 5.6, 3.8, 4.1, 8.9, 2.5, 6.3],
        'revenue': [1000, 800, 600, 400, 1200, 1100, 900, 700, 500, 1300],
        'waste_kg': [1.2, 0.8, 2.5, 0.3, 1.8, 1.0, 0.7, 3.0, 0.4, 2.2],
        'operational_anomaly': [False, False, True, False, False, False, False, True, False, False],
        'theoretical_unit_cost': [4.2, 2.8, 1.5, 0.8, 2.2, 4.2, 2.8, 1.5, 0.8, 2.2]
    }
    return pd.DataFrame(data)

def test_basic_pdf():
    """Test de base de la génération PDF"""
    print("🧪 TEST BASIC PDF")
    print("=" * 40)
    
    if not PDF_AVAILABLE:
        print("1. Installation des dépendances...")
        os.system("pip install reportlab")
        try:
            from pdf_reporter import COSPDFReporter
            print("✅ reportlab installé")
        except:
            print("❌ Échec de l'installation")
            return False
    
    print("\n2. Création des données de test...")
    df = create_minimal_test_data()
    restaurant_df = pd.DataFrame({
        'restaurant_name': ['KFC Paris Louvre', 'KFC Lyon Part-Dieu'],
        'city': ['Paris', 'Lyon']
    })
    
    print(f"   {len(df)} transactions créées")
    
    print("\n3. Création du générateur PDF...")
    reporter = COSPDFReporter(output_dir="reports_simple")
    
    print("\n4. Génération d'un rapport test...")
    try:
        report_path = reporter.generate_weekly_report(df, restaurant_df, week_number=1)
        
        if report_path and os.path.exists(report_path):
            print(f"✅ SUCCÈS ! Rapport généré: {report_path}")
            print(f"   Taille: {os.path.getsize(report_path):,} octets")
            return True
        else:
            print("❌ ÉCHEC : Aucun rapport généré")
            return False
    except Exception as e:
        print(f"❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_pdf()
    if success:
        print("\n🎉 Test PDF réussi !")
        print("\nProchaine étape: intégrer au dashboard")
    else:
        print("\n❌ Test PDF échoué")
    sys.exit(0 if success else 1)