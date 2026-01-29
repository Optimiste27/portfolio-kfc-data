
# EXEMPLE D'INTÉGRATION DU MODÈLE DANS LE DASHBOARD

import pickle
import pandas as pd
import numpy as np

def load_cos_model():
    """Charge le modèle COS pour prédiction"""
    with open('models/cos_final_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']['names']
    
    return model, scaler, features

def predict_risk(new_data):
    """Prédit le risque COS pour de nouvelles données"""
    model, scaler, features = load_cos_model()
    
    # Préparer les données
    X_new = new_data[features]
    X_scaled = scaler.transform(X_new)
    
    # Prédictions
    risk_proba = model.predict_proba(X_scaled)[:, 1]
    risk_class = model.predict(X_scaled)
    
    return {
        'risk_probability': risk_proba,
        'risk_class': risk_class,
        'risk_level': ['Faible' if p < 0.4 else 'Moyen' if p < 0.7 else 'Élevé' for p in risk_proba]
    }

# Utilisation dans le dashboard
def update_risk_dashboard():
    """Met à jour le dashboard avec les nouvelles prédictions"""
    # 1. Charger les nouvelles données
    new_data = load_daily_data()
    
    # 2. Faire les prédictions
    predictions = predict_risk(new_data)
    
    # 3. Mettre à jour les visualisations
    update_risk_map(predictions)
    update_risk_charts(predictions)
    
    # 4. Envoyer les alertes si nécessaire
    send_alerts_if_needed(predictions)
