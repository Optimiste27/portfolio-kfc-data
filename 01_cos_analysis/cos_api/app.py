"""
Application Flask principale pour l'API COS
"""

import os
from flask import Flask
from flask_cors import CORS
from flask_restx import Api
from flask_jwt_extended import JWTManager
from prometheus_flask_exporter import PrometheusMetrics
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Import des blueprints/namespaces
from api.health import health_ns
from api.predictions import predict_ns

# Import des utilitaires
from utils.logging import setup_logging
from models.predictor import COSPredictor

def create_app(config_name='default'):
    """Factory pour créer l'application Flask"""
    
    # Créer l'application
    app = Flask(__name__)
    
    # Charger la configuration
    from config import config
    app.config.from_object(config[config_name])
    
    # Initialiser le logging
    setup_logging(app)
    
    # Initialiser CORS
    CORS(app, resources={
        r"/*": {
            "origins": app.config['CORS_ORIGINS'],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Initialiser JWT (optionnel)
    jwt = JWTManager(app)
    
    # Initialiser l'API REST
    authorizations = {
        'Bearer Auth': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': "Format: 'Bearer {token}'"
        }
    }
    
    api = Api(
        app,
        version='1.0.0',
        title='COS Prediction API',
        description='API de prédiction des risques COS pour KFC',
        doc='/docs',  # Swagger UI à /docs
        authorizations=authorizations,
        security='Bearer Auth'
    )
    
    # Ajouter les namespaces
    api.add_namespace(health_ns, path='/api/health')
    api.add_namespace(predict_ns, path='/api/predict')
    
    # Initialiser Prometheus Metrics
    metrics = PrometheusMetrics(app)
    
    # Initialiser le prédicteur
    try:
        model_path = app.config.get('MODEL_PATH')
        app.predictor = COSPredictor(model_path)
        app.logger.info("✅ Prédicteur initialisé avec succès")
    except Exception as e:
        app.logger.error(f"❌ Erreur initialisation prédicteur: {e}")
        # En production, on peut choisir de ne pas démarrer
        if app.config.get('ENV') == 'production':
            raise
    
    # Compteur de requêtes
    app.request_count = 0
    
    # Middleware pour logging des requêtes
    @app.before_request
    def before_request():
        app.logger.info(
            f"Request: {request.method} {request.path}",
            extra={'ip': request.remote_addr}
        )
    
    @app.after_request
    def after_request(response):
        app.logger.info(
            f"Response: {response.status_code}",
            extra={
                'ip': request.remote_addr,
                'status': response.status_code,
                'method': request.method,
                'path': request.path
            }
        )
        return response
    
    # Routes de base
    @app.route('/')
    def index():
        """Page d'accueil"""
        return {
            "message": "COS Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "documentation": "/docs",
                "health": "/api/health",
                "predict": "/api/predict/single",
                "batch_predict": "/api/predict/batch",
                "model_info": "/api/predict/model-info"
            }
        }
    
    @app.route('/favicon.ico')
    def favicon():
        return '', 204
    
    app.logger.info(f"✅ Application initialisée en mode {config_name}")
    return app

if __name__ == '__main__':
    # Démarrer l'application
    app = create_app(os.getenv('FLASK_ENV', 'default'))
    
    # Démarrer le serveur
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    app.logger.info(f"🚀 Démarrage serveur sur {host}:{port}")
    app.run(host=host, port=port, debug=app.config.get('DEBUG', False))