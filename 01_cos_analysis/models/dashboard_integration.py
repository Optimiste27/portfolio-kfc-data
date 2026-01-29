"""
EXEMPLE D'UTILISATION DU MODÈLE COS DANS LE DASHBOARD
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

class COSRiskPredictor:
    """Classe pour prédire les risques COS dans le dashboard"""
    
    def __init__(self, model_path='models/cos_clean_classifier.pkl'):
        """Charge le modèle entraîné"""
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.features = self.model_data['features']
        self.threshold = self.model_data['threshold']
        self.performance = self.model_data['performance']
        
        print(f"✅ Modèle COS chargé (version {self.model_data.get('version', '3.0')})")
        print(f"   • Performance: F1={self.performance['test']['f1']:.3f}")
        print(f"   • Features: {len(self.features)}")
        print(f"   • Seuil risque: > {self.threshold:.2f}%")
    
    def prepare_features(self, restaurant_data):
        """Prépare les features à partir des données du restaurant"""
        # Créer un DataFrame avec les features attendues
        features_df = pd.DataFrame(index=[0])
        
        for feature in self.features:
            if feature in restaurant_data:
                features_df[feature] = restaurant_data[feature]
            else:
                # Valeur par défaut si la feature manque
                # Pour les régions, mettre à 0 (pas cette région)
                if feature.startswith('region_'):
                    features_df[feature] = 0
                else:
                    features_df[feature] = 0
        
        return features_df
    
    def predict_risk(self, restaurant_data):
        """Prédit le risque COS pour un restaurant"""
        try:
            # Préparer les features
            X = self.prepare_features(restaurant_data)
            
            # Normaliser
            X_scaled = self.scaler.transform(X)
            
            # Prédiction
            risk_probability = self.model.predict_proba(X_scaled)[:, 1][0]
            risk_class = self.model.predict(X_scaled)[0]
            
            # Niveau de risque
            if risk_probability > 0.7:
                risk_level = "ÉLEVÉ"
                action = "🚨 Alerte immédiate - Audit requis"
                color = "red"
            elif risk_probability > 0.4:
                risk_level = "MODÉRÉ"
                action = "⚠️  Surveillance renforcée"
                color = "orange"
            else:
                risk_level = "FAIBLE"
                action = "✅ Monitoring standard"
                color = "green"
            
            # Identifier la région principale
            region = "Inconnue"
            for col in self.features:
                if col.startswith('region_') and col in restaurant_data and restaurant_data[col] == 1:
                    region = col.replace('region_', '')
                    break
            
            return {
                'restaurant_id': restaurant_data.get('restaurant_id', 'N/A'),
                'restaurant_name': restaurant_data.get('restaurant_name', 'N/A'),
                'region': region,
                'risk_probability': float(risk_probability),
                'risk_class': int(risk_class),
                'risk_level': risk_level,
                'recommended_action': action,
                'color': color,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'confidence': 'HAUTE' if risk_probability > 0.7 else 'MOYENNE' if risk_probability > 0.4 else 'BASSE'
            }
            
        except Exception as e:
            print(f"❌ Erreur de prédiction: {e}")
            return {
                'restaurant_id': restaurant_data.get('restaurant_id', 'N/A'),
                'error': str(e),
                'risk_level': 'ERREUR',
                'color': 'gray'
            }
    
    def predict_batch(self, restaurants_data):
        """Prédit les risques pour plusieurs restaurants"""
        predictions = []
        
        for restaurant_data in restaurants_data:
            prediction = self.predict_risk(restaurant_data)
            predictions.append(prediction)
        
        # Trier par niveau de risque (élevé d'abord)
        predictions.sort(key=lambda x: x.get('risk_probability', 0), reverse=True)
        
        return predictions

# EXEMPLE D'UTILISATION DANS LE DASHBOARD
def dashboard_example():
    """Exemple d'intégration dans Streamlit ou autre dashboard"""
    
    # 1. Initialiser le prédicteur
    predictor = COSRiskPredictor('models/cos_clean_classifier.pkl')
    
    # 2. Exemple de données (à remplacer par vos vraies données)
    daily_data = [
        {
            'restaurant_id': 1,
            'restaurant_name': 'KFC Marseille Vieux Port',
            'region_Marseille': 1,
            'region_Paris': 0,
            'region_Lyon': 0,
            'operational_anomaly': 1,
            'anomaly_count_7d': 2,
            'qsp_score': 7.5,
            'units_sold': 180,
            'selling_price': 12.99,
            'month': 7,
            'day_of_week': 2,
            'is_weekend': 0,
        },
        {
            'restaurant_id': 2,
            'restaurant_name': 'KFC Paris Centre',
            'region_Marseille': 0,
            'region_Paris': 1,
            'region_Lyon': 0,
            'operational_anomaly': 0,
            'anomaly_count_7d': 0,
            'qsp_score': 9.0,
            'units_sold': 120,
            'selling_price': 11.99,
            'month': 7,
            'day_of_week': 2,
            'is_weekend': 0,
        },
    ]
    
    # 3. Faire les prédictions
    predictions = predictor.predict_batch(daily_data)
    
    # 4. Afficher les résultats (format dashboard)
    print("\n🎯 TABLEAU DE BORD - RISQUES COS")
    print("="*60)
    
    high_risk = [p for p in predictions if p.get('risk_level') == 'ÉLEVÉ']
    medium_risk = [p for p in predictions if p.get('risk_level') == 'MODÉRÉ']
    low_risk = [p for p in predictions if p.get('risk_level') == 'FAIBLE']
    
    print(f"\n🚨 RISQUES ÉLEVÉS ({len(high_risk)}):")
    for pred in high_risk:
        print(f"   • {pred['restaurant_name']} - {pred['risk_probability']:.1%}")
        print(f"     {pred['recommended_action']}")
    
    print(f"\n⚠️  RISQUES MODÉRÉS ({len(medium_risk)}):")
    for pred in medium_risk:
        print(f"   • {pred['restaurant_name']} - {pred['risk_probability']:.1%}")
    
    print(f"\n✅ RISQUES FAIBLES ({len(low_risk)}):")
    for pred in low_risk:
        print(f"   • {pred['restaurant_name']}")
    
    return predictions

if __name__ == "__main__":
    # Exemple d'exécution
    dashboard_example()
