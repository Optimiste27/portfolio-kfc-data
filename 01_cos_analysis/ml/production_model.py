"""
ENTRAÎNEMENT ML FINAL SANS FEATURES RÉGIONALES
Version simplifiée pour production
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class ProductionModelTrainer:
    def __init__(self, dataset_path='../data/processed/ml_dataset_OPTIMIZED.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.models = {}
        self.results = {}
        
    def load_and_clean_data(self):
        """Charge et nettoie les données - supprime les features régionales"""
        print("📥 Chargement et nettoyage des données...")
        self.df = pd.read_csv(self.dataset_path)
        
        # Supprimer les features de région (trop dominantes)
        region_cols = [col for col in self.df.columns if col.startswith('region_')]
        category_cols = [col for col in self.df.columns if col.startswith('category_')]
        
        print(f"✅ Dataset original: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        print(f"🗑️  Suppression de {len(region_cols)} features de région")
        print(f"🗑️  Suppression de {len(category_cols)} features de catégorie")
        
        # Garder seulement les features métier
        business_features = [
            # Temporelles
            'month', 'day_of_week', 'is_weekend', 'week_of_year',
            'is_month_start', 'is_month_end',
            
            # Métier
            'selling_price', 'units_sold', 'qsp_score',
            'price_norm_by_category', 'price_to_units_ratio',
            
            # Historiques
            'rest_units_ma_14', 'rest_units_ma_30',
            'rest_units_std_14', 'rest_units_std_30',
            'rest_qsp_ma_14', 'rest_qsp_ma_30',
            
            # Interactions
            'rest_performance_30d', 'prod_popularity_in_rest',
            'units_trend_7d', 'qsp_trend_14d',
            
            # Cible
            'gap_percentage'
        ]
        
        # Garder seulement les colonnes qui existent
        existing_features = [col for col in business_features if col in self.df.columns]
        self.df = self.df[existing_features]
        
        print(f"📊 Dataset final: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        print(f"🎯 Target: gap_percentage (mean={self.df['gap_percentage'].mean():.2f}%, std={self.df['gap_percentage'].std():.2f}%)")
        
        return self.df
    
    def train_production_model(self):
        """Entraîne un modèle simple pour production"""
        print("\n🚀 ENTRAÎNEMENT DU MODÈLE DE PRODUCTION")
        print("="*60)
        
        # Séparer features et target
        X = self.df.drop(columns=['gap_percentage'])
        y = self.df['gap_percentage']
        
        print(f"📊 Données:")
        print(f"  • Features: {X.shape[1]}")
        print(f"  • Samples: {X.shape[0]}")
        
        # TimeSeriesSplit simple
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Modèles de production (simples et robustes)
        self.models = {
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,  # Très shallow pour éviter overfitting
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=20,
                random_state=42
            )
        }
        
        # Résultats
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔧 {name}:")
            
            r2_scores = []
            rmse_scores = []
            mae_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Normalisation
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement
                model.fit(X_train_scaled, y_train)
                
                # Prédiction
                y_pred = model.predict(X_test_scaled)
                
                # Métriques
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                r2_scores.append(r2)
                rmse_scores.append(rmse)
                mae_scores.append(mae)
            
            # Stocker les résultats
            results[name] = {
                'model': model,
                'avg_r2': np.mean(r2_scores),
                'avg_rmse': np.mean(rmse_scores),
                'avg_mae': np.mean(mae_scores),
                'std_r2': np.std(r2_scores),
                'r2_scores': r2_scores,
                'rmse_scores': rmse_scores
            }
            
            print(f"  • R²: {np.mean(r2_scores):+.3f} ± {np.std(r2_scores):.3f}")
            print(f"  • RMSE: {np.mean(rmse_scores):.3f}%")
            print(f"  • MAE: {np.mean(mae_scores):.3f}%")
        
        self.results = results
        return results
    
    def select_best_production_model(self):
        """Sélectionne le meilleur modèle pour production"""
        print("\n🏆 SÉLECTION DU MODÈLE DE PRODUCTION")
        print("="*60)
        
        # Critère : RMSE moyen le plus bas avec R² > 0
        ranking = []
        for name, result in self.results.items():
            if result['avg_r2'] > 0:  # Seulement modèles avec R² positif
                ranking.append({
                    'Modèle': name,
                    'RMSE Moyen': result['avg_rmse'],
                    'R² Moyen': result['avg_r2'],
                    'Stabilité': result['std_r2'],
                    'R² par fold': result['r2_scores']
                })
        
        if not ranking:
            print("❌ Aucun modèle avec R² > 0")
            return None
        
        ranking_df = pd.DataFrame(ranking)
        ranking_df = ranking_df.sort_values('RMSE Moyen')
        
        print("\n📊 CLASSEMENT:")
        print(ranking_df[['Modèle', 'R² Moyen', 'RMSE Moyen', 'Stabilité']].to_string(index=False))
        
        best_model_name = ranking_df.iloc[0]['Modèle']
        best_result = self.results[best_model_name]
        
        print(f"\n🎯 MODÈLE SÉLECTIONNÉ: {best_model_name}")
        print(f"   • R²: {best_result['avg_r2']:.3f}")
        print(f"   • RMSE: {best_result['avg_rmse']:.3f}%")
        print(f"   • MAE: {best_result['avg_mae']:.3f}%")
        print(f"   • Stabilité: {best_result['std_r2']:.3f}")
        
        # Interprétation business
        print(f"\n💡 INTERPRÉTATION BUSINESS:")
        rmse = best_result['avg_rmse']
        r2 = best_result['avg_r2']
        
        print(f"   • Précision: ±{rmse:.2f}% sur le gap COS")
        print(f"   • Capacité prédictive: {r2*100:.1f}% de la variance expliquée")
        
        if rmse < 1.0:
            print(f"   ✅ EXCELLENT: Précision suffisante pour actions correctives")
        elif rmse < 1.5:
            print(f"   ✅ BON: Utile pour identification des tendances")
        elif rmse < 2.0:
            print(f"   🟡 ACCEPTABLE: À combiner avec expertise métier")
        
        return best_model_name
    
    def save_production_model(self, model_name):
        """Sauvegarde le modèle pour production"""
        print(f"\n💾 Sauvegarde du modèle de production...")
        
        import os
        os.makedirs('../models', exist_ok=True)
        os.makedirs('../reports', exist_ok=True)
        
        model = self.results[model_name]['model']
        
        # Créer package pour production
        model_package = {
            'model': model,
            'model_name': model_name,
            'model_type': str(type(model)).split("'")[1],
            'metrics': self.results[model_name],
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_count': self.df.shape[1] - 1,
            'features_list': list(self.df.drop(columns=['gap_percentage']).columns),
            'target_info': {
                'mean': float(self.df['gap_percentage'].mean()),
                'std': float(self.df['gap_percentage'].std()),
                'min': float(self.df['gap_percentage'].min()),
                'max': float(self.df['gap_percentage'].max())
            },
            'version': '1.0.0',
            'description': 'Modèle de prédiction du COS (Cost of Sales) pour KFC'
        }
        
        # Sauvegarder modèle
        model_path = '../models/cos_production_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        
        # Sauvegarder rapport de production
        self.save_production_report(model_name, model_path)
        
        return model_path
    
    def save_production_report(self, model_name, model_path):
        """Génère un rapport pour production"""
        report = f"""
{'='*70}
📋 RAPPORT DE PRODUCTION - MODÈLE COS KFC
{'='*70}

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏆 Modèle: {model_name}
📁 Fichier: {model_path}

📊 PERFORMANCE DU MODÈLE
{'─'*40}
• R² Moyen: {self.results[model_name]['avg_r2']:.3f}
• RMSE Moyen: {self.results[model_name]['avg_rmse']:.3f}%
• MAE Moyen: {self.results[model_name]['avg_mae']:.3f}%
• Stabilité (std R²): {self.results[model_name]['std_r2']:.3f}

🎯 INTERPRÉTATION BUSINESS
{'─'*40}
• Le modèle prédit le gap COS avec une erreur moyenne de ±{self.results[model_name]['avg_rmse']:.2f}%
• Il explique {self.results[model_name]['avg_r2']*100:.1f}% de la variabilité du COS
• Performance par fold: {self.results[model_name]['r2_scores']}

📈 RECOMMANDATIONS D'UTILISATION
{'─'*40}
1. UTILISATION OPÉRATIONNELLE:
   • Identifier les restaurants à risque (gap prédit > 5%)
   • Planifier les actions correctives 1-2 jours à l'avance
   • Optimiser les commandes de matières premières

2. INTERPRÉTATION DES PRÉDICTIONS:
   • Une prédiction de 4.5% ±1.2% signifie un gap probable entre 3.3% et 5.7%
   • Se concentrer sur les tendances plutôt que les valeurs exactes
   • Combiner avec l'expertise des managers régionaux

3. SURVEILLANCE:
   • Recalibrer le modèle tous les 3 mois
   • Surveiller la dérive des performances
   • Ajouter de nouvelles données saisonnières

🔧 CARACTÉRISTIQUES TECHNIQUES
{'─'*40}
• Features utilisées: {self.df.shape[1] - 1}
• Période d'entraînement: Jan 2024 - Jun 2024
• Validation: TimeSeriesSplit (3 folds)
• Données: {len(self.df):,} échantillons

🚀 INTÉGRATION DASHBOARD
{'─'*40}
1. Charger le modèle: pickle.load(open('{model_path}', 'rb'))
2. Préparer les données: mêmes features que l'entraînement
3. Normaliser: StandardScaler comme pendant l'entraînement
4. Prédire: model.predict(scaled_features)
5. Afficher: gap prédit ± marge d'erreur

📞 SUPPORT
{'─'*40}
• Pour les questions techniques: équipe data science
• Pour l'interprétation métier: directeurs régionaux
• Mise à jour du modèle: tous les trimestres

{'='*70}
✅ MODÈLE PRÊT POUR LA PRODUCTION
{'='*70}
"""
        
        report_path = '../reports/production_model_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Rapport de production: {report_path}")
        
        # Aussi sauvegarder en JSON pour le dashboard
        import json
        dashboard_info = {
            'model_name': model_name,
            'model_path': model_path,
            'performance': {
                'r2': float(self.results[model_name]['avg_r2']),
                'rmse': float(self.results[model_name]['avg_rmse']),
                'mae': float(self.results[model_name]['avg_mae'])
            },
            'features': list(self.df.drop(columns=['gap_percentage']).columns),
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        json_path = '../models/model_dashboard_info.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_info, f, indent=2)
        
        print(f"✅ Info dashboard: {json_path}")

def main():
    """Fonction principale pour entraînement production"""
    print("="*70)
    print("🏭 ENTRAÎNEMENT MODÈLE DE PRODUCTION - COS KFC")
    print("="*70)
    
    trainer = ProductionModelTrainer()
    
    try:
        # 1. Charger et nettoyer
        print("\n[1/4] 📊 PRÉPARATION DES DONNÉES")
        df = trainer.load_and_clean_data()
        
        # 2. Entraînement
        print("\n[2/4] 🚀 ENTRAÎNEMENT")
        results = trainer.train_production_model()
        
        # 3. Sélection
        print("\n[3/4] 🏆 SÉLECTION")
        best_model = trainer.select_best_production_model()
        
        if best_model:
            # 4. Sauvegarde
            print("\n[4/4] 💾 SAUVEGARDE")
            model_path = trainer.save_production_model(best_model)
            
            print("\n" + "="*70)
            print("✅ MODÈLE DE PRODUCTION PRÊT !")
            print("="*70)
            
            print(f"\n🎯 RÉSULTATS:")
            print(f"  • Modèle: {best_model}")
            print(f"  • Performance: R²={trainer.results[best_model]['avg_r2']:.3f}")
            print(f"  • Précision: ±{trainer.results[best_model]['avg_rmse']:.2f}%")
            
            print(f"\n📁 ARTEFACTS:")
            print(f"  • Modèle: ../models/cos_production_model.pkl")
            print(f"  • Rapport: ../reports/production_model_report.txt")
            print(f"  • Dashboard info: ../models/model_dashboard_info.json")
            
            print(f"\n🚀 INTÉGRATION IMMÉDIATE POSSIBLE DANS LE DASHBOARD!")
            
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()