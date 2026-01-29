"""
Endpoints de prédiction pour le modèle COS
"""

from flask_restx import Namespace, Resource, fields
from flask import request, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import jwt_required, get_jwt_identity
import time

predict_ns = Namespace('predict', description='Endpoints de prédiction COS')

# Modèles Swagger/OpenAPI
prediction_model = predict_ns.model('PredictionRequest', {
    'restaurant_id': fields.String(required=True, description='ID du restaurant'),
    'restaurant_name': fields.String(description='Nom du restaurant'),
    'operational_anomaly': fields.Float(description='Anomalie opérationnelle (0/1)'),
    'anomaly_count_7d': fields.Float(description='Nombre d\'anomalies sur 7 jours'),
    'anomaly_count_30d': fields.Float(description='Nombre d\'anomalies sur 30 jours'),
    'qsp_score': fields.Float(description='Score QSP (0-10)'),
    'units_sold': fields.Float(description='Unités vendues'),
    'selling_price': fields.Float(description='Prix de vente'),
    'month': fields.Float(description='Mois (1-12)'),
    'day_of_week': fields.Float(description='Jour de la semaine (0-6)'),
    'is_weekend': fields.Float(description='Weekend (0/1)'),
    'region_Paris': fields.Float(description='Région Paris (0/1)'),
    'region_Lyon': fields.Float(description='Région Lyon (0/1)'),
    'region_Marseille': fields.Float(description='Région Marseille (0/1)'),
    'region_Toulouse': fields.Float(description='Région Toulouse (0/1)'),
    'region_Lille': fields.Float(description='Région Lille (0/1)'),
    'region_Bordeaux': fields.Float(description='Région Bordeaux (0/1)'),
    'region_Nantes': fields.Float(description='Région Nantes (0/1)'),
    'region_Strasbourg': fields.Float(description='Région Strasbourg (0/1)'),
    'region_Rennes': fields.Float(description='Région Rennes (0/1)'),
    'region_Nice': fields.Float(description='Région Nice (0/1)')
})

batch_model = predict_ns.model('BatchPredictionRequest', {
    'restaurants': fields.List(fields.Nested(prediction_model), required=True, description='Liste des restaurants')
})

prediction_response = predict_ns.model('PredictionResponse', {
    'success': fields.Boolean(description='Succès de la prédiction'),
    'prediction_id': fields.String(description='ID de la prédiction'),
    'timestamp': fields.String(description='Timestamp de la prédiction'),
    'restaurant_id': fields.String(description='ID du restaurant'),
    'restaurant_name': fields.String(description='Nom du restaurant'),
    'region': fields.String(description='Région détectée'),
    'risk_probability': fields.Float(description='Probabilité de risque (0-1)'),
    'risk_class': fields.Integer(description='Classe de risque (0/1)'),
    'risk_level': fields.String(description='Niveau de risque (HIGH/MEDIUM/LOW)'),
    'color': fields.String(description='Couleur pour le dashboard'),
    'recommended_action': fields.String(description='Action recommandée'),
    'threshold_exceeded': fields.Boolean(description='Seuil dépassé'),
    'model_version': fields.String(description='Version du modèle'),
    'warnings': fields.List(fields.String, description='Avertissements')
})

batch_response = predict_ns.model('BatchPredictionResponse', {
    'success': fields.Boolean(description='Succès global'),
    'total_predictions': fields.Integer(description='Nombre total de prédictions'),
    'successful_predictions': fields.Integer(description='Prédictions réussies'),
    'failed_predictions': fields.Integer(description='Prédictions échouées'),
    'risk_distribution': fields.Raw(description='Distribution des risques'),
    'predictions': fields.List(fields.Nested(prediction_response)),
    'errors': fields.List(fields.Raw, description='Erreurs détaillées'),
    'timestamp': fields.String(description='Timestamp')
})

# Initialiser le rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute", "10 per second"]
)

@predict_ns.route('/single')
class SinglePrediction(Resource):
    """Prédiction pour un seul restaurant"""
    
    @predict_ns.expect(prediction_model)
    @predict_ns.marshal_with(prediction_response)
    @limiter.limit("60 per minute")
    # @jwt_required()  # Décommenter pour activer l'authentification
    def post(self):
        """Prédire le risque COS pour un restaurant"""
        start_time = time.time()
        
        # Récupérer les données
        data = request.get_json()
        
        # Valider les données
        from utils.validators import InputValidator
        is_valid, errors, cleaned_data = InputValidator.validate_restaurant_data(data)
        
        if not is_valid:
            predict_ns.abort(400, "Validation error", errors=errors)
        
        # Faire la prédiction
        try:
            predictor = current_app.predictor
            result = predictor.predict_single(cleaned_data)
            
            # Log de la prédiction
            current_app.logger.info(
                f"Prediction completed",
                extra={
                    'restaurant_id': cleaned_data.get('restaurant_id'),
                    'risk_level': result.get('risk_level'),
                    'risk_probability': result.get('risk_probability'),
                    'processing_time': time.time() - start_time
                }
            )
            
            # Incrémenter le compteur de requêtes
            if not hasattr(current_app, 'request_count'):
                current_app.request_count = 0
            current_app.request_count += 1
            
            return result
            
        except Exception as e:
            current_app.logger.error(f"Prediction error: {e}")
            predict_ns.abort(500, "Internal server error")

@predict_ns.route('/batch')
class BatchPrediction(Resource):
    """Prédiction par lot pour plusieurs restaurants"""
    
    @predict_ns.expect(batch_model)
    @predict_ns.marshal_with(batch_response)
    @limiter.limit("10 per minute")
    # @jwt_required()  # Décommenter pour activer l'authentification
    def post(self):
        """Prédire le risque COS pour plusieurs restaurants"""
        start_time = time.time()
        
        # Récupérer les données
        data = request.get_json()
        
        if 'restaurants' not in data:
            predict_ns.abort(400, "Missing 'restaurants' field")
        
        restaurants = data['restaurants']
        
        # Valider les données
        from utils.validators import InputValidator
        is_valid, errors, cleaned_data = InputValidator.validate_batch_data(restaurants)
        
        if not is_valid and len(errors) > 10:
            predict_ns.abort(400, "Too many validation errors", errors=errors[:10])
        
        # Faire la prédiction
        try:
            predictor = current_app.predictor
            result = predictor.predict_batch(cleaned_data)
            
            # Log de la prédiction batch
            current_app.logger.info(
                f"Batch prediction completed",
                extra={
                    'total_predictions': result.get('total_predictions'),
                    'successful_predictions': result.get('successful_predictions'),
                    'processing_time': time.time() - start_time
                }
            )
            
            return result
            
        except Exception as e:
            current_app.logger.error(f"Batch prediction error: {e}")
            predict_ns.abort(500, "Internal server error")

@predict_ns.route('/model-info')
class ModelInfo(Resource):
    """Informations sur le modèle"""
    
    @limiter.limit("30 per minute")
    def get(self):
        """Retourne les informations du modèle chargé"""
        try:
            predictor = current_app.predictor
            return predictor.get_model_info()
        except Exception as e:
            predict_ns.abort(500, f"Error getting model info: {e}")

@predict_ns.route('/features')
class FeaturesList(Resource):
    """Liste des features attendues"""
    
    @limiter.limit("30 per minute")
    def get(self):
        """Retourne la liste des features attendues par le modèle"""
        try:
            predictor = current_app.predictor
            return {
                "features": predictor.features,
                "count": len(predictor.features),
                "required": True
            }
        except Exception as e:
            predict_ns.abort(500, f"Error getting features: {e}")