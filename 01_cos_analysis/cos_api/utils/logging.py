"""
Configuration du logging pour l'API
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Formateur de logs en JSON"""
    
    def format(self, record):
        log_object = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_object['request_id'] = record.request_id
        
        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_object)

def setup_logging(app):
    """Configure le logging pour l'application Flask"""
    
    # Niveau de log
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if app.config.get('ENV') == 'production':
        # Format JSON en production
        formatter = JSONFormatter()
    else:
        # Format lisible en développement
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    
    # Handler pour les fichiers (rotation)
    if app.config.get('LOG_FILE'):
        file_handler = RotatingFileHandler(
            app.config['LOG_FILE'],
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Configurer le logger racine
    logging.getLogger().setLevel(log_level)
    logging.getLogger().addHandler(console_handler)
    
    # Réduire le bruit des logs externes
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    app.logger = logging.getLogger(__name__)