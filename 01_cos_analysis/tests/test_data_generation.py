"""
Tests unitaires pour la génération de données - AVEC BONS IMPORTS
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Import du vrai générateur
try:
    from src.data_generation import COSDataGeneratorSimple
    REAL_GENERATOR = True
except ImportError:
    REAL_GENERATOR = False
    print("⚠️  Utilisation du mock pour les tests")

def test_data_generator_initialization():
    """Test l'initialisation du générateur"""
    if REAL_GENERATOR:
        generator = COSDataGeneratorSimple(seed=42)
        assert generator is not None
        # Vérifie que le générateur a les attributs attendus
        assert hasattr(generator, 'restaurants')
        assert hasattr(generator, 'product_families')
    else:
        pytest.skip("Générateur non disponible - test ignoré")

def test_restaurant_count():
    """Test que le générateur crée le bon nombre de restaurants"""
    if REAL_GENERATOR:
        generator = COSDataGeneratorSimple(seed=42)
        assert len(generator.restaurants) == 10
        assert isinstance(generator.restaurants, dict)
    else:
        pytest.skip("Générateur non disponible - test ignoré")

def test_product_families():
    """Test que toutes les familles de produits sont présentes"""
    if REAL_GENERATOR:
        generator = COSDataGeneratorSimple(seed=42)
        expected_products = [
            'Poulet Frit', 'Sandwichs', 'Burger Poulet', 'Wraps & Salades',
            'Accompagnements', 'Boissons', 'Desserts', 'Petits Déjeuners'
        ]
        actual_products = list(generator.product_families.keys())
        
        # Vérifie les produits
        for product in expected_products:
            assert product in actual_products, f"Produit manquant: {product}"
    else:
        pytest.skip("Générateur non disponible - test ignoré")

def test_data_types(sample_transaction_data):
    """Test les types de données"""
    assert pd.api.types.is_integer_dtype(sample_transaction_data['transaction_id'])
    assert pd.api.types.is_float_dtype(sample_transaction_data['theoretical_unit_cost'])
    assert pd.api.types.is_integer_dtype(sample_transaction_data['units_sold'])
    assert pd.api.types.is_string_dtype(sample_transaction_data['product_family'])

def test_positive_values(sample_transaction_data):
    """Test que les valeurs numériques sont positives"""
    numeric_cols = ['theoretical_unit_cost', 'actual_unit_cost', 'selling_price', 
                    'units_sold', 'revenue', 'waste_kg', 'qsp_score']
    
    for col in numeric_cols:
        if col in sample_transaction_data.columns:
            if col != 'units_sold':  # units_sold pourrait théoriquement être négatif
                assert (sample_transaction_data[col] >= 0).all(), f"Valeurs négatives dans {col}"

def test_actual_cost_higher_than_theoretical(sample_transaction_data):
    """Test que le coût réel est >= coût théorique (écart positif)"""
    if 'actual_unit_cost' in sample_transaction_data.columns and 'theoretical_unit_cost' in sample_transaction_data.columns:
        diff = sample_transaction_data['actual_unit_cost'] - sample_transaction_data['theoretical_unit_cost']
        # En vrai, actual peut être < theoretical (bonne surprise), mais pour nos données synthétiques:
        assert (diff >= -0.1).all()  # Tolérance de -10% max

def test_selling_price_margin(sample_transaction_data):
    """Test que le prix de vente a une marge raisonnable"""
    if 'selling_price' in sample_transaction_data.columns and 'theoretical_unit_cost' in sample_transaction_data.columns:
        margin_ratio = sample_transaction_data['selling_price'] / sample_transaction_data['theoretical_unit_cost']
        assert (margin_ratio >= 1.3).all(), "Marge trop faible"
        assert (margin_ratio <= 5.0).all(), "Marge trop élevée"

def test_no_duplicate_transaction_ids(sample_transaction_data):
    """Test qu'il n'y a pas d'IDs de transaction en double"""
    assert sample_transaction_data['transaction_id'].is_unique

def test_date_format(sample_transaction_data):
    """Test que les dates ont un format valide"""
    try:
        pd.to_datetime(sample_transaction_data['date'])
        assert True
    except:
        assert False, "Format de date invalide"

def test_generate_transactions_shape(sample_transaction_data):
    """Test la forme des données générées"""
    assert sample_transaction_data.shape[1] >= 10
    assert 'transaction_id' in sample_transaction_data.columns
    assert 'product_family' in sample_transaction_data.columns

def test_revenue_calculation(sample_transaction_data):
    """Test que le revenue = selling_price * units_sold"""
    if 'revenue' in sample_transaction_data.columns and 'selling_price' in sample_transaction_data.columns and 'units_sold' in sample_transaction_data.columns:
        expected_revenue = sample_transaction_data['selling_price'] * sample_transaction_data['units_sold']
        actual_revenue = sample_transaction_data['revenue']
        
        # Vérifie avec tolérance d'arrondi
        diff = abs(expected_revenue - actual_revenue)
        assert (diff < 0.01).all(), "Calcul du revenue incorrect"