"""
Classe de prédiction pour le modèle COS
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class COSPredictor:
    """Prédicteur de risques COS"""
    
    def __init__(self, model_path: str):
        """Initialise le prédicteur avec le chemin du modèle"""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.features = None
        self.metadata = None
        self.loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle depuis le fichier pickle"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.metadata = {
                'threshold': model_data.get('threshold', 5.79),
                'version': model_data.get('version', '3.0.0'),
                'training_date': model_data.get('training_date', 'unknown'),
                'performance': model_data.get('performance', {})
            }
            
            self.loaded = True
            logger.info(f"✅ Modèle chargé: {self.metadata['version']}")
            logger.info(f"   Features: {len(self.features)}")
            logger.info(f"   Performance: {self.metadata['performance'].get('test', {}).get('f1', 0):.3f}")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        if not self.loaded:
            return {"error": "Modèle non chargé"}
        
        return {
            "status": "loaded",
            "version": self.metadata['version'],
            "training_date": self.metadata['training_date'],
            "features_count": len(self.features),
            "features": self.features,
            "threshold": self.metadata['threshold'],
            "performance": self.metadata['performance']
        }
    
    def validate_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Valide et prépare les features d'entrée"""
        validated = {}
        warnings = []
        
        # Pour chaque feature attendue
        for feature in self.features:
            if feature in input_data:
                value = input_data[feature]
                
                # Validation des types
                try:
                    # Convertir en float si possible
                    validated[feature] = float(value)
                except (ValueError, TypeError):
                    warnings.append(f"Feature {feature}: valeur invalide '{value}'")
                    # Valeur par défaut pour les features numériques
                    if feature.startswith('region_') or feature.startswith('category_'):
                        validated[feature] = 0.0
                    else:
                        validated[feature] = 0.0
            else:
                # Feature manquante
                if feature.startswith('region_') or feature.startswith('category_'):
                    validated[feature] = 0.0  # Valeur par défaut pour one-hot
                else:
                    validated[feature] = 0.0  # Valeur par défaut
                warnings.append(f"Feature manquante: {feature}")
        
        # Features supplémentaires non attendues
        extra_features = set(input_data.keys()) - set(self.features)
        if extra_features:
            warnings.append(f"Features supplémentaires ignorées: {list(extra_features)}")
        
        return {
            "features": validated,
            "warnings": warnings,
            "valid": len(warnings) == 0
        }
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prédit le risque pour un seul restaurant"""
        if not self.loaded:
            return {"error": "Modèle non chargé", "risk_level": "ERROR"}
        
        try:
            # Validation
            validation = self.validate_features(input_data)
            
            if not validation["valid"] and len(validation["warnings"]) > 5:
                return {
                    "error": "Trop d'erreurs de validation",
                    "warnings": validation["warnings"][:5],
                    "risk_level": "ERROR"
                }
            
            # Préparation des données
            features_df = pd.DataFrame([validation["features"]])
            features_scaled = self.scaler.transform(features_df)
            
            # Prédiction
            risk_probability = self.model.predict_proba(features_scaled)[0, 1]
            risk_class = self.model.predict(features_scaled)[0]
            
            # Détermination du niveau de risque
            if risk_probability > 0.7:
                risk_level = "HIGH"
                color = "#dc3545"  # Rouge
                action = "🚨 Audit immédiat requis"
            elif risk_probability > 0.4:
                risk_level = "MEDIUM"
                color = "#ffc107"  # Orange
                action = "⚠️ Surveillance renforcée"
            else:
                risk_level = "LOW"
                color = "#28a745"  # Vert
                action = "✅ Monitoring standard"
            
            # Identifier la région principale
            region = "Unknown"
            for feature in self.features:
                if feature.startswith('region_') and validation["features"].get(feature, 0) == 1:
                    region = feature.replace('region_', '').replace('_', ' ')
                    break
            
            return {
                "success": True,
                "prediction_id": f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "restaurant_id": input_data.get("restaurant_id", "unknown"),
                "restaurant_name": input_data.get("restaurant_name", "unknown"),
                "region": region,
                "risk_probability": float(risk_probability),
                "risk_class": int(risk_class),
                "risk_level": risk_level,
                "color": color,
                "recommended_action": action,
                "threshold_exceeded": risk_probability > (self.metadata['threshold'] / 100),
                "model_version": self.metadata['version'],
                "warnings": validation["warnings"]
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            return {
                "error": str(e),
                "success": False,
                "risk_level": "ERROR"
            }
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prédit le risque pour plusieurs restaurants"""
        if not self.loaded:
            return {"error": "Modèle non chargé", "predictions": []}
        
        predictions = []
        errors = []
        
        for i, input_data in enumerate(input_data_list):
            try:
                prediction = self.predict_single(input_data)
                predictions.append(prediction)
                
                if not prediction.get("success", False):
                    errors.append({
                        "index": i,
                        "restaurant_id": input_data.get("restaurant_id", f"item_{i}"),
                        "error": prediction.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                errors.append({
                    "index": i,
                    "restaurant_id": input_data.get("restaurant_id", f"item_{i}"),
                    "error": str(e)
                })
        
        # Statistiques
        risk_counts = {}
        for pred in predictions:
            if pred.get("success", False):
                level = pred.get("risk_level", "UNKNOWN")
                risk_counts[level] = risk_counts.get(level, 0) + 1
        
        return {
            "success": True,
            "total_predictions": len(input_data_list),
            "successful_predictions": len(predictions),
            "failed_predictions": len(errors),
            "risk_distribution": risk_counts,
            "predictions": predictions,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }