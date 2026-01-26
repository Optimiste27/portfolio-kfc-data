#!/usr/bin/env python
"""
Script pour exécuter tous les tests unitaires - VERSION AMÉLIORÉE
"""

import sys
import os
import subprocess
import argparse

def setup_environment():
    """Configure l'environnement Python pour les tests"""
    # Ajoute le chemin src au PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Vérifie que le fichier existe
    data_gen_path = os.path.join(src_path, "data_generation.py")
    if not os.path.exists(data_gen_path):
        print(f"⚠️  Fichier non trouvé: {data_gen_path}")
        print("   Vérifiez que le fichier src/data_generation.py existe")
        return False
    
    print(f"✅ Chemin src: {src_path}")
    print(f"✅ Fichier data_generation.py: {'TROUVÉ' if os.path.exists(data_gen_path) else 'NON TROUVÉ'}")
    
    return True

def run_tests(test_dir="tests", verbose=False):
    """Exécute tous les tests unitaires"""
    print("🧪 LANCEMENT DES TESTS UNITAIRES")
    print("=" * 60)
    
    # Configuration de l'environnement
    if not setup_environment():
        return False
    
    # Vérifie que le dossier tests existe
    if not os.path.exists(test_dir):
        print(f"❌ Dossier {test_dir} non trouvé")
        print("   Créez-le avec: mkdir tests")
        return False
    
    # Construit la commande pytest
    cmd = [sys.executable, "-m", "pytest", test_dir]
    
    if verbose:
        cmd.append("-v")
    
    # Exécute un test d'import d'abord
    print("\n🔍 TEST D'IMPORT:")
    try:
        from src.data_generation import COSDataGeneratorSimple
        print("✅ Import COSDataGeneratorSimple: SUCCÈS")
    except ImportError as e:
        print(f"❌ Import COSDataGeneratorSimple: ÉCHEC - {e}")
        print("\n💡 SOLUTION: Vérifiez que:")
        print("   1. Vous êtes dans le dossier 01_cos_analysis")
        print("   2. Le fichier src/data_generation.py existe")
        print("   3. Le fichier a la classe COSDataGeneratorSimple")
        return False
    
    try:
        # Exécute les tests
        print(f"\n📁 Dossier de tests: {test_dir}")
        print(f"🐍 Python: {sys.executable}")
        print("🔧 Exécution des tests...\n")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Affiche la sortie
        print("📋 RÉSULTATS DES TESTS:")
        print("=" * 60)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  ERREURS:")
            print(result.stderr)
        
        print("=" * 60)
        
        if result.returncode == 0:
            print("✅ TOUS LES TESTS PASSENT !")
            return True
        else:
            print(f"❌ {result.returncode} TESTS ÉCHOUÉS")
            return False
            
    except FileNotFoundError:
        print("❌ pytest non installé. Installez-le avec: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution des tests: {e}")
        return False

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Exécute les tests unitaires du projet COS")
    parser.add_argument("--test-dir", default="tests", help="Dossier contenant les tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")
    parser.add_argument("--quick", action="store_true", help="Test rapide d'import seulement")
    
    args = parser.parse_args()
    
    if args.quick:
        # Test rapide d'import seulement
        print("🔍 TEST RAPIDE D'IMPORT")
        print("=" * 40)
        success = setup_environment()
        if success:
            print("\n✅ Environnement configuré avec succès")
            print("💡 Pour exécuter tous les tests: python run_tests.py")
        else:
            print("\n❌ Problème de configuration")
        sys.exit(0 if success else 1)
    
    success = run_tests(args.test_dir, args.verbose)
    
    # Résumé
    print("\n📊 RÉSUMÉ DE LA QUALITÉ:")
    print("=" * 60)
    if success:
        print("✅ Fiabilité: Excellente - Tous les tests passent")
        print("✅ Maintenabilité: Code testé, facile à modifier")
        print("✅ Robustesse: Gestion des cas limites testée")
        print("✅ Documentation: Tests comme documentation vivante")
    else:
        print("⚠️  Fiabilité: À améliorer - Tests échoués")
        print("⚠️  Maintenabilité: Risque de régression")
        print("💡 Recommandation: Corriger les tests avant de continuer")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()