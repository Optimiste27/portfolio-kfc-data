"""
Configuration des tests unitaires - AVEC BONS IMPORTS
"""


# AJOUTE CES LIGNES AU DÉBUT
import matplotlib
matplotlib.use('Agg')  # Backend non interactif

import pytest
import pandas as pd
import numpy as np
import sys
import os


# Ajoute les chemins nécessaires
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import du vrai générateur
try:
    from src.data_generation import COSDataGeneratorSimple
    GENERATOR_AVAILABLE = True
    print("✅ Import réussi: COSDataGeneratorSimple depuis src.data_generation")
except ImportError as e:
    print(f"⚠️  Import échoué: {e}")
    
    # Mock pour les tests
    class COSDataGeneratorSimple:
        """Mock pour les tests quand le vrai générateur n'est pas disponible"""
        def __init__(self, seed=42):
            self.seed = seed
            self.TARGET_GAP = 0.04  # 4%
            self.restaurants = {
                1: {"name": "KFC Paris Louvre", "city": "Paris", "profile": "performant"},
                2: {"name": "KFC Lyon Part-Dieu", "city": "Lyon", "profile": "excellent"},
                # ... etc pour les 10 restaurants
            }
            self.product_families = {
                "Poulet Frit": {"base_cost": 4.20, "margin_multiplier": 2.1},
                "Sandwichs": {"base_cost": 2.80, "margin_multiplier": 2.3},
                # ... etc
            }
    
    GENERATOR_AVAILABLE = False

# Fixtures de données de test
@pytest.fixture
def sample_transaction_data():
    """Données de test pour les transactions"""
    data = pd.DataFrame({
        'transaction_id': [10001, 10002, 10003, 10004],
        'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'restaurant_id': [1, 1, 2, 2],
        'restaurant_name': ['KFC Paris Louvre', 'KFC Paris Louvre', 'KFC Lyon Part-Dieu', 'KFC Lyon Part-Dieu'],
        'product_family': ['Poulet Frit', 'Sandwichs', 'Poulet Frit', 'Boissons'],
        'theoretical_unit_cost': [4.20, 2.80, 4.20, 0.80],
        'actual_unit_cost': [4.41, 2.94, 4.62, 0.84],  # +5%, +5%, +10%, +5%
        'selling_price': [9.24, 6.44, 9.24, 2.40],     # Marge ~2.2x
        'units_sold': [10, 15, 8, 20],
        'revenue': [92.40, 96.60, 73.92, 48.00],
        'waste_kg': [0.5, 0.2, 0.6, 0.1],
        'qsp_score': [0.85, 0.90, 0.75, 0.88]
    })
    return data

@pytest.fixture
def sample_restaurant_data():
    """Données de test pour les restaurants"""
    return pd.DataFrame({
        'restaurant_id': [1, 2],
        'restaurant_name': ['KFC Paris Louvre', 'KFC Lyon Part-Dieu'],
        'city': ['Paris', 'Lyon'],
        'performance_category': ['Bon', 'À améliorer'],
        'target_gap_percentage': [3.0, 7.0],
        'manager_experience': [36, 12],
        'team_size': [25, 22],
        'last_audit_score': [0.88, 0.72]
    })

@pytest.fixture
def expected_kpis():
    """Résultats attendus pour les KPI"""
    return {
        'theoretical_cos': [0.4545, 0.4348, 0.4545, 0.3333],
        'actual_cos': [0.4773, 0.4565, 0.5000, 0.3500],
        'cos_gap': [0.0228, 0.0217, 0.0455, 0.0167],
        'gap_percentage': [5.02, 5.00, 10.00, 5.00]
    }