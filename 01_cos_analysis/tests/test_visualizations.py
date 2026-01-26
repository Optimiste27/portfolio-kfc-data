"""
Tests unitaires pour les visualisations
"""
# AJOUTE AU DÉBUT DU FICHIER
import matplotlib
matplotlib.use('Agg')  # Pas besoin d'interface graphique pour les tests
import matplotlib.pyplot as plt

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_create_gap_distribution_plot():
    """Test la création d'un histogramme de distribution"""
    # Crée des données de test
    np.random.seed(42)
    gap_data = np.random.normal(4.0, 2.0, 1000)  # 1000 écarts autour de 4%
    
    # Crée le plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gap_data, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(4.0, color='red', linestyle='--', label='Cible (4.0%)')
    ax.set_xlabel('Écart COS (%)')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des écarts COS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Vérifie que le plot a été créé
    assert fig is not None
    assert ax is not None
    assert len(ax.patches) == 30  # 30 bins
    
    plt.close(fig)

def test_create_restaurant_comparison_chart(sample_transaction_data):
    """Test la création d'un graphique de comparaison par restaurant"""
    # Calcule les KPI
    df = sample_transaction_data.copy()
    df['theoretical_material_cost'] = df['theoretical_unit_cost'] * df['units_sold']
    df['actual_material_cost'] = df['actual_unit_cost'] * df['units_sold']
    df['theoretical_cos'] = df['theoretical_material_cost'] / df['revenue']
    df['actual_cos'] = df['actual_material_cost'] / df['revenue']
    df['gap_percentage'] = ((df['actual_cos'] - df['theoretical_cos']) / df['theoretical_cos']) * 100
    
    # Agrège par restaurant
    restaurant_stats = df.groupby('restaurant_name').agg({
        'gap_percentage': 'mean',
        'revenue': 'sum'
    }).round(2)
    
    # Crée le plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x > 6 else 'orange' if x > 4 else 'green' 
              for x in restaurant_stats['gap_percentage']]
    
    bars = ax.bar(restaurant_stats.index, restaurant_stats['gap_percentage'], 
                  color=colors, edgecolor='black')
    
    ax.set_xlabel('Restaurant')
    ax.set_ylabel('Écart moyen (%)')
    ax.set_title('Performance par restaurant')
    ax.axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='Cible (4.0%)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Vérifications
    assert fig is not None
    assert len(bars) == len(restaurant_stats)
    
    plt.close(fig)

def test_create_product_analysis_chart(sample_transaction_data):
    """Test la création d'un graphique par produit"""
    # Agrège par produit
    product_stats = sample_transaction_data.groupby('product_family').agg({
        'revenue': 'sum',
        'units_sold': 'sum'
    }).round(2)
    
    # Crée le plot
    fig, ax = plt.subplots(figsize=(10, 6))
    product_stats_sorted = product_stats.sort_values('revenue', ascending=True)
    
    bars = ax.barh(product_stats_sorted.index, product_stats_sorted['revenue'],
                   edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Chiffre d\'affaires (€)')
    ax.set_title('Performance par famille de produit')
    
    # Ajoute les valeurs
    for i, (idx, row) in enumerate(product_stats_sorted.iterrows()):
        ax.text(row['revenue'] + 5, i, f"{row['revenue']} €", va='center')
    
    # Vérifications
    assert fig is not None
    assert len(bars) == len(product_stats)
    
    plt.close(fig)

def test_plot_labels_and_titles():
    """Test que les labels et titres sont corrects"""
    fig, ax = plt.subplots()
    
    # Test data
    x = [1, 2, 3]
    y = [4, 5, 6]
    
    ax.plot(x, y)
    ax.set_xlabel('Temps (jours)')
    ax.set_ylabel('Écart COS (%)')
    ax.set_title('Évolution des écarts sur la période')
    ax.grid(True)
    
    # Vérifie les labels
    assert ax.get_xlabel() == 'Temps (jours)'
    assert ax.get_ylabel() == 'Écart COS (%)'
    assert ax.get_title() == 'Évolution des écarts sur la période'
    assert ax.xaxis.get_gridlines()[0].get_visible()  # Grid visible
    
    plt.close(fig)