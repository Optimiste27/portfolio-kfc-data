#!/usr/bin/env python3
"""
Lancement du dashboard COMPLET avec authentification
"""

import subprocess
import os

print("=" * 70)
print("🔐 DASHBOARD COS KFC - VERSION COMPLÈTE AVEC AUTHENTIFICATION")
print("=" * 70)
print()
print("🎯 IDENTIFIANTS DE CONNEXION :")
print("  • admin    / admin123     → Accès COMPLET (toutes les fonctionnalités)")
print("  • manager  / manager123   → Gestion + PDFs (pas d'administration)")
print("  • viewer   / viewer123    → Consultation seule (données limitées)")
print()
print("📊 FONCTIONNALITÉS PAR RÔLE :")
print("  👑 ADMIN : Toutes les fonctionnalités + Administration")
print("  👨‍💼 MANAGER : Dashboard complet + PDFs (sans admin)")
print("  👁️ VIEWER : Consultation limitée (100 transactions max)")
print("=" * 70)
print()

# Lancer le dashboard
subprocess.run(["streamlit", "run", "app_with_auth.py"])