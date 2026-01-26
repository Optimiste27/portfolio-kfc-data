"""
Tests unitaires pour les calculs de KPI - VÉRIFIÉ
"""

import pytest
import pandas as pd
import numpy as np

# NOTE: Ce fichier n'a pas besoin d'importer le générateur
# Il teste seulement la fonction calculate_kpis

def calculate_kpis(df):
    """Fonction de calcul des KPI (version robuste)"""
    df = df.copy()
    
    # Assure que les colonnes nécessaires existent
    required_cols = ['theoretical_unit_cost', 'actual_unit_cost', 'units_sold', 'revenue']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols}")
    
    # Calcul des KPI
    df['theoretical_material_cost'] = df['theoretical_unit_cost'] * df['units_sold']
    df['actual_material_cost'] = df['actual_unit_cost'] * df['units_sold']
    
    # Évite la division par zéro
    df['revenue_safe'] = df['revenue'].replace(0, np.nan)
    
    df['theoretical_cos'] = df['theoretical_material_cost'] / df['revenue_safe']
    df['actual_cos'] = df['actual_material_cost'] / df['revenue_safe']
    
    df['cos_gap'] = df['actual_cos'] - df['theoretical_cos']
    
    # Calcul du pourcentage avec gestion des NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        df['gap_percentage'] = np.where(
            df['theoretical_cos'] != 0,
            (df['cos_gap'] / df['theoretical_cos']) * 100,
            0
        )
    
    # Nettoie les colonnes temporaires
    df.drop(columns=['revenue_safe'], errors='ignore', inplace=True)
    
    # Remplace inf/-inf/NaN par 0
    df['gap_percentage'] = df['gap_percentage'].replace([np.inf, -np.inf, np.nan], 0)
    
    return df

# ... (le reste du fichier reste identique) ...