"""
Tests pour l'API COS
"""

import pytest
import json
from app import create_app
from unittest.mock import patch

@pytest.fixture
def app():
    """Créer une application de test"""
    app = create_app('testing')
    yield app

@pytest.fixture
def client(app):
    """Client de test"""
    return app.test_client()

@pytest.fixture
def auth_header():
    """Header d'authentification (simulé)"""
    return {'Authorization': 'Bearer test-token'}

def test_health_check(client):
    """Test du endpoint de santé"""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_model_info(client):
    """Test des informations du modèle"""
    response = client.get('/api/predict/model-info')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'version' in data

def test_single_prediction(client):
    """Test d'une prédiction simple"""
    test_data = {
        'restaurant_id': 'TEST_001',
        'restaurant_name': 'KFC Test',
        'region_Paris': 1,
        'operational_anomaly': 0,
        'qsp_score': 8.5,
        'units_sold': 150,
        'selling_price': 12.99,
        'month': 7,
        'day_of_week': 2,
        'is_weekend': 0
    }
    
    response = client.post(
        '/api/predict/single',
        json=test_data,
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'risk_level' in data
    assert 'risk_probability' in data

def test_batch_prediction(client):
    """Test d'une prédiction par lot"""
    test_data = {
        'restaurants': [
            {
                'restaurant_id': 'TEST_001',
                'region_Paris': 1,
                'qsp_score': 8.5
            },
            {
                'restaurant_id': 'TEST_002',
                'region_Marseille': 1,
                'qsp_score': 6.0
            }
        ]
    }
    
    response = client.post(
        '/api/predict/batch',
        json=test_data,
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'predictions' in data
    assert len(data['predictions']) == 2

def test_invalid_input(client):
    """Test avec des données invalides"""
    test_data = {
        'restaurant_id': 123,
        'qsp_score': 'not-a-number'  # Type invalide
    }
    
    response = client.post(
        '/api/predict/single',
        json=test_data,
        content_type='application/json'
    )
    
    # Devrait retourner 400 ou 422
    assert response.status_code in [400, 422]

def test_missing_required_field(client):
    """Test avec champ requis manquant"""
    test_data = {
        'qsp_score': 8.5
        # restaurant_id manquant
    }
    
    response = client.post(
        '/api/predict/single',
        json=test_data,
        content_type='application/json'
    )
    
    assert response.status_code == 400

def test_rate_limiting(client):
    """Test du rate limiting"""
    test_data = {'restaurant_id': 'TEST', 'region_Paris': 1}
    
    # Faire plusieurs requêtes rapidement
    for i in range(15):
        response = client.post('/api/predict/single', json=test_data)
    
    # La 11ème devrait être limitée (10 par seconde)
    assert response.status_code == 429