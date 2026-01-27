#!/usr/bin/env python3
"""
Script de lancement du dashboard sécurisé
"""

import subprocess
import time

def main():
    print("=" * 60)
    print("🔐 DASHBOARD COS KFC - AUTHENTIFICATION")
    print("=" * 60)
    print()
    print("IDENTIFIANTS DE TEST:")
    print("  admin    / admin123")
    print("  manager  / manager123")
    print("  viewer   / viewer123")
    print()
    print("RÔLES:")
    print("  • Admin: Accès complet + administration")
    print("  • Manager: Gestion + consultation")
    print("  • Viewer: Consultation seule")
    print("=" * 60)
    print()
    
    print("🔄 Lancement du dashboard...")
    time.sleep(1)
    
    # Lancer Streamlit
    subprocess.run(["streamlit", "run", "dashboard_secure.py"])

if __name__ == "__main__":
    main()