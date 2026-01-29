# ml/model_training_clean.py
"""
Entraînement ML avec dataset propre
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

class CleanModelTrainer:
    def __init__(self, dataset_path='../data/processed/ml_dataset_clean.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Charge le dataset propre"""
        print("📥 Chargement du dataset propre...")
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"✅ Dataset chargé: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        print(f"🎯 Target stats: mean={self.df['gap_percentage'].mean():.2f}%, std={self.df['gap_percentage'].std():.2f}%")
        
        return self.df
    
    def prepare_data(self, time_based_split=True):
        """Prépare les données pour l'entraînement"""
        print("\n📊 Préparation des données...")
        
        # Séparer features et target
        X = self.df.drop(columns=['gap_percentage'])
        y = self.df['gap_percentage']
        
        if time_based_split and len(self.df) > 1000:
            # Split temporel (80/20)
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            print(f"  • Split temporel (80/20)")
            print(f"  • Train: {len(X_train):,} échantillons")
            print(f"  • Test: {len(X_test):,} échantillons")
        else:
            # Split aléatoire
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=not time_based_split
            )
            print(f"  • Split aléatoire (80/20)")
            print(f"  • Train: {len(X_train):,} échantillons")
            print(f"  • Test: {len(X_test):,} échantillons")
        
        # Normaliser
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print(f"  • Features: {X_train.shape[1]}")
        print(f"  • Normalisation: Oui")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Entraîne et évalue les modèles"""
        print("\n🚀 ENTRAÎNEMENT DES MODÈLES")
        print("="*60)
        
        # Initialiser les modèles
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.01, random_state=42, max_iter=5000),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        if XGB_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        
        if LGBM_AVAILABLE:
            self.models['LightGBM'] = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        # Entraînement et évaluation
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n🔧 {name}")
            
            try:
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Métriques
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Stocker
                self.results[name] = {
                    'model': model,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'y_pred_test': y_pred_test
                }
                
                print(f"  Train: MAE={train_mae:.3f}%, RMSE={train_rmse:.3f}%, R²={train_r2:.3f}")
                print(f"  Test:  MAE={test_mae:.3f}%, RMSE={test_rmse:.3f}%, R²={test_r2:.3f}")
                
            except Exception as e:
                print(f"  ❌ Erreur: {str(e)[:50]}")
                self.results[name] = None
        
        return self.results
    
    def select_best_model(self):
        """Sélectionne le meilleur modèle"""
        print("\n🏆 SÉLECTION DU MEILLEUR MODÈLE")
        print("="*60)
        
        # Créer DataFrame des résultats
        results_df = []
        
        for name, result in self.results.items():
            if result is not None:
                results_df.append({
                    'Modèle': name,
                    'RMSE Test': result['test_rmse'],
                    'MAE Test': result['test_mae'],
                    'R² Test': result['test_r2'],
                    'Diff Train/Test': abs(result['test_rmse'] - result['train_rmse']) / result['train_rmse']
                })
        
        if not results_df:
            print("❌ Aucun modèle entraîné avec succès")
            return None, None
        
        results_df = pd.DataFrame(results_df)
        results_df = results_df.sort_values('RMSE Test')
        
        print("\n📊 CLASSEMENT:")
        print(results_df[['Modèle', 'RMSE Test', 'MAE Test', 'R² Test']].to_string(index=False))
        
        best_model_name = results_df.iloc[0]['Modèle']
        best_result = self.results[best_model_name]
        
        print(f"\n🎯 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   • RMSE: {best_result['test_rmse']:.3f}%")
        print(f"   • MAE: {best_result['test_mae']:.3f}%")
        print(f"   • R²: {best_result['test_r2']:.3f}")
        
        # Interprétation réaliste
        rmse = best_result['test_rmse']
        if rmse < 0.5:
            print(f"   🎉 Excellente précision!")
        elif rmse < 1.0:
            print(f"   ✅ Très bonne précision")
        elif rmse < 1.5:
            print(f"   🟡 Bonne précision")
        elif rmse < 2.0:
            print(f"   🟠 Précision acceptable")
        else:
            print(f"   🔴 Précision à améliorer")
        
        return best_model_name, results_df
    
    def save_model(self, best_model_name, scaler):
        """Sauvegarde le modèle"""
        print(f"\n💾 Sauvegarde: {best_model_name}")
        
        import os
        os.makedirs('../models', exist_ok=True)
        
        best_model = self.results[best_model_name]['model']
        
        # Package à sauvegarder
        model_package = {
            'model': best_model,
            'model_name': best_model_name,
            'scaler': scaler,
            'metrics': {
                'test_rmse': self.results[best_model_name]['test_rmse'],
                'test_mae': self.results[best_model_name]['test_mae'],
                'test_r2': self.results[best_model_name]['test_r2']
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': f"ml_dataset_clean.csv - {len(self.df)} samples"
        }
        
        # Sauvegarder
        model_path = '../models/best_cos_model_clean.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        return model_path

def main():
    """Fonction principale"""
    print("="*70)
    print("🧠 ENTRAÎNEMENT ML RÉALISTE - SANS DATA LEAKAGE")
    print("="*70)
    
    trainer = CleanModelTrainer()
    
    try:
        # 1. Charger les données
        print("\n[1/4] 📥 CHARGEMENT")
        df = trainer.load_data()
        
        # 2. Préparer les données
        print("\n[2/4] 📊 PRÉPARATION")
        X_train, X_test, y_train, y_test, scaler = trainer.prepare_data(time_based_split=True)
        
        # 3. Entraîner et évaluer
        print("\n[3/4] 🚀 ENTRAÎNEMENT")
        results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # 4. Sélectionner et sauvegarder
        print("\n[4/4] 🏆 SÉLECTION")
        best_model_name, results_df = trainer.select_best_model()
        
        if best_model_name:
            model_path = trainer.save_model(best_model_name, scaler)
            
            print("\n" + "="*70)
            print("✅ ENTRAÎNEMENT RÉALISTE TERMINÉ !")
            print("="*70)
            print(f"\n🎯 RÉSULTATS RÉALISTES:")
            print(f"  • Meilleur modèle: {best_model_name}")
            print(f"  • Précision (RMSE): {trainer.results[best_model_name]['test_rmse']:.3f}%")
            print(f"  • Qualité (R²): {trainer.results[best_model_name]['test_r2']:.3f}")
            print(f"  • Modèle sauvegardé: {model_path}")
            print(f"\n🚀 Prêt pour l'intégration dashboard!")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()