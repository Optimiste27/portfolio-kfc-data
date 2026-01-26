"""
Vérifie la structure du projet
"""

import os
import sys

def check_project_structure():
    print("🔍 VÉRIFICATION DE LA STRUCTURE DU PROJET")
    print("=" * 50)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Chemins à vérifier
    paths_to_check = [
        ("src/", "dossier"),
        ("src/data_generation.py", "fichier"),
        ("tests/", "dossier"),
        ("tests/__init__.py", "fichier"),
        ("tests/conftest.py", "fichier"),
    ]
    
    all_ok = True
    
    for path, type_ in paths_to_check:
        full_path = os.path.join(base_dir, path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"{status} {path} ({type_})")
        
        if not exists and type_ == "fichier":
            # Vérifie le contenu du fichier
            print(f"   Contenu attendu dans {path}:")
            if path == "src/data_generation.py":
                print("   - class COSDataGenerator:")
                print("   - def __init__(self, seed=42):")
                print("   - self.restaurants = {...}")
            elif path == "tests/conftest.py":
                print("   - import pytest, pandas, numpy")
                print("   - @pytest.fixture decorators")
        
        if not exists:
            all_ok = False
    
    # Vérifie le contenu de data_generation.py
    data_gen_path = os.path.join(base_dir, "src/data_generation.py")
    if os.path.exists(data_gen_path):
        print("\n📄 CONTENU DE src/data_generation.py:")
        with open(data_gen_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class COSDataGenerator" in content:
                print("✅ Contient class COSDataGenerator")
            else:
                print("❌ NE contient PAS class COSDataGenerator")
                all_ok = False
    
    print(f"\n{'✅ STRUCTURE OK' if all_ok else '❌ STRUCTURE INCOMPLÈTE'}")
    return all_ok

if __name__ == "__main__":
    check_project_structure()