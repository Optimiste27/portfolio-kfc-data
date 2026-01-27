#!/usr/bin/env python3
"""
Lancement du dashboard sécurisé
"""

import os
import sys
import subprocess

def main():
    """Lance le dashboard avec authentification"""
    
    # Vérification des dépendances
    try:
        import jwt
        import streamlit
        print("✅ Dépendances OK")
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("Installez: pip install pyjwt streamlit")
        return
    
    # Lancement
    print("🔒 Lancement du Dashboard Sécurisé COS KFC")
    print("=" * 50)
    print("Accès par défaut:")
    print("  • admin / admin123")
    print("  • manager / manager123")
    print("  • viewer / viewer123")
    print("=" * 50)
    
    subprocess.run(["streamlit", "run", "app_secure.py"])

if __name__ == "__main__":
    main()