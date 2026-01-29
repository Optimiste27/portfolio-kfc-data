"""
Validateurs pour les données d'entrée
"""

import re
from typing import Dict, Any, List, Tuple
from datetime import datetime

class InputValidator:
    """Validateur des données d'entrée"""
    
    @staticmethod
    def validate_restaurant_data(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Valide les données d'un restaurant"""
        errors = []
        cleaned = data.copy()
        
        # Validation du restaurant_id
        restaurant_id = data.get('restaurant_id')
        if not restaurant_id:
            errors.append("restaurant_id est requis")
        elif not isinstance(restaurant_id, (int, str)):
            errors.append("restaurant_id doit être un entier ou une chaîne")
        else:
            cleaned['restaurant_id'] = str(restaurant_id)
        
        # Validation du restaurant_name
        restaurant_name = data.get('restaurant_name', '').strip()
        if not restaurant_name:
            errors.append("restaurant_name est recommandé")
        elif len(restaurant_name) > 100:
            errors.append("restaurant_name trop long (max 100 caractères)")
        else:
            cleaned['restaurant_name'] = restaurant_name
        
        # Validation des features numériques
        numeric_features = [
            'operational_anomaly', 'anomaly_count_7d', 'anomaly_count_30d',
            'qsp_score', 'units_sold', 'selling_price', 'month', 'day_of_week',
            'is_weekend', 'week_of_year', 'is_month_start', 'is_month_end'
        ]
        
        for feature in numeric_features:
            if feature in data:
                try:
                    value = float(data[feature])
                    if feature in ['operational_anomaly', 'is_weekend', 'is_month_start', 'is_month_end']:
                        # Features binaires
                        cleaned[feature] = 1 if value > 0.5 else 0
                    elif feature == 'qsp_score':
                        # QSP entre 0 et 10
                        if 0 <= value <= 10:
                            cleaned[feature] = value
                        else:
                            errors.append(f"{feature} doit être entre 0 et 10")
                            cleaned[feature] = max(0, min(value, 10))
                    elif feature == 'month':
                        # Mois entre 1 et 12
                        if 1 <= value <= 12:
                            cleaned[feature] = int(value)
                        else:
                            errors.append(f"{feature} doit être entre 1 et 12")
                            cleaned[feature] = max(1, min(int(value), 12))
                    elif feature == 'day_of_week':
                        # Jour entre 0 et 6
                        if 0 <= value <= 6:
                            cleaned[feature] = int(value)
                        else:
                            errors.append(f"{feature} doit être entre 0 et 6")
                            cleaned[feature] = max(0, min(int(value), 6))
                    else:
                        # Autres features numériques
                        cleaned[feature] = value
                except (ValueError, TypeError):
                    errors.append(f"{feature} doit être numérique")
                    cleaned[feature] = 0.0
        
        # Validation des features one-hot (régions, catégories)
        one_hot_features = []
        for key in data.keys():
            if key.startswith('region_') or key.startswith('category_'):
                one_hot_features.append(key)
        
        for feature in one_hot_features:
            value = data[feature]
            try:
                num_value = float(value)
                cleaned[feature] = 1 if num_value > 0.5 else 0
            except (ValueError, TypeError):
                errors.append(f"{feature} doit être 0 ou 1")
                cleaned[feature] = 0
        
        return len(errors) == 0, errors, cleaned
    
    @staticmethod
    def validate_batch_data(data_list: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
        """Valide une liste de données de restaurants"""
        errors = []
        cleaned_list = []
        
        if not isinstance(data_list, list):
            return False, ["Les données doivent être une liste"], []
        
        if len(data_list) > 1000:
            errors.append("Trop d'éléments (max 1000)")
            data_list = data_list[:1000]
        
        for i, data in enumerate(data_list):
            is_valid, item_errors, cleaned = InputValidator.validate_restaurant_data(data)
            if not is_valid:
                for err in item_errors:
                    errors.append(f"Item {i}: {err}")
            cleaned_list.append(cleaned)
        
        return len(errors) == 0, errors, cleaned_list