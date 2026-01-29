"""
Endpoints de santé et monitoring
"""

from flask_restx import Namespace, Resource
from flask import current_app
import psutil
import os
from datetime import datetime

health_ns = Namespace('health', description='Endpoints de santé')

@health_ns.route('')
class HealthCheck(Resource):
    """Vérification de la santé de l'API"""
    
    def get(self):
        """Retourne le statut de santé de l'API"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "COS Prediction API",
            "version": current_app.config.get('API_VERSION', '1.0.0')
        }

@health_ns.route('/detailed')
class DetailedHealth(Resource):
    """Vérification détaillée de la santé"""
    
    def get(self):
        """Retourne une vérification détaillée de la santé"""
        # Informations système
        system_info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "process_memory_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        }
        
        # Statut des dépendances
        dependencies = {
            "model_loaded": current_app.predictor.loaded if hasattr(current_app, 'predictor') else False,
            "database": "connected" if hasattr(current_app, 'db') else "not_configured"
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system": system_info,
            "dependencies": dependencies,
            "environment": current_app.config.get('ENV', 'unknown')
        }

@health_ns.route('/ready')
class ReadinessCheck(Resource):
    """Vérification de readiness (prêt à servir)"""
    
    def get(self):
        """Vérifie si l'API est prête à servir des requêtes"""
        # Vérifier que le modèle est chargé
        if not hasattr(current_app, 'predictor') or not current_app.predictor.loaded:
            return {"status": "not_ready", "reason": "Model not loaded"}, 503
        
        # Vérifier la mémoire
        if psutil.virtual_memory().percent > 90:
            return {"status": "not_ready", "reason": "High memory usage"}, 503
        
        return {"status": "ready"}, 200

@health_ns.route('/metrics')
class Metrics(Resource):
    """Métriques Prometheus (simplifié)"""
    
    def get(self):
        """Retourne des métriques au format Prometheus"""
        # Ces métriques seraient normalement exposées via prometheus-flask-exporter
        metrics = []
        
        # Métriques système
        metrics.append(f"# HELP system_cpu_percent CPU usage percentage")
        metrics.append(f"# TYPE system_cpu_percent gauge")
        metrics.append(f"system_cpu_percent {psutil.cpu_percent()}")
        
        metrics.append(f"# HELP system_memory_percent Memory usage percentage")
        metrics.append(f"# TYPE system_memory_percent gauge")
        metrics.append(f"system_memory_percent {psutil.virtual_memory().percent}")
        
        # Métriques application
        if hasattr(current_app, 'request_count'):
            metrics.append(f"# HELP api_requests_total Total number of API requests")
            metrics.append(f"# TYPE api_requests_total counter")
            metrics.append(f"api_requests_total {current_app.request_count}")
        
        return "\n".join(metrics), 200, {'Content-Type': 'text/plain'}