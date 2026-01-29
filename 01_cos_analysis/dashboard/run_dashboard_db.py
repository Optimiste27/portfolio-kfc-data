#!/usr/bin/env python3
"""
Lancement du dashboard avec base de données SQLite
"""

import subprocess
import os

print("=" * 70)
print("🏪 DASHBOARD COS KFC - VERSION BASE DE DONNÉES SQLite")
print("=" * 70)
print()
print("🎯 FONCTIONNALITÉS NOUVELLES:")
print("  • Base de données SQLite (performances ×10)")
print("  • Requêtes SQL optimisées")
print("  • Gestion transactions 14,400+")
print("  • Authentification en base")
print("  • Backup automatique")
print()
print("🔐 IDENTIFIANTS:")
print("  • admin / admin123     → Accès complet + administration")
print("  • manager / manager123 → Dashboard + PDFs")
print("  • viewer / viewer123   → Consultation seule")
print("=" * 70)
print()

# Vérifier que la base de données existe
if not os.path.exists("cos_kfc.db"):
    print("⚠️  Attention: La base de données 'cos_kfc.db' n'existe pas")
    print("   Exécutez d'abord: python data_migration.py")
    choice = input("Voulez-vous exécuter la migration maintenant? (o/n): ")
    if choice.lower() == 'o':
        import data_migration
        data_migration.main()

# Lancer le dashboard
subprocess.run(["streamlit", "run", "dashboard_db.py"])