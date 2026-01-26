#!/usr/bin/env python3
"""
Script de lancement du dashboard Streamlit
"""

import os
import sys
import subprocess
import webbrowser
from threading import Timer

def open_browser():
    """Ouvre le navigateur automatiquement"""
    webbrowser.open_new("http://localhost:8501")

def main():
    """Lance le dashboard Streamlit"""
    print("🚀 Lancement du Dashboard COS KFC")
    print("=" * 50)
    
    # Crée les dossiers nécessaires
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    # Ouvre le navigateur après 2 secondes
    Timer(2, open_browser).start()
    
    # Lance Streamlit
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()