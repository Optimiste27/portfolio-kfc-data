"""
Entraînement ML OPTIMISÉ avec dataset propre et anti-overfitting
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

class OptimizedModelTrainer:
    def __init__(self, dataset_path='../data/processed/ml_dataset_OPTIMIZED.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Charge le dataset optimisé"""
        print("📥 Chargement du dataset optimisé...")
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"✅ Dataset chargé: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        print(f"🎯 Target stats: mean={self.df['gap_percentage'].mean():.2f}%, std={self.df['gap_percentage'].std():.2f}%")
        
        return self.df
    
    def prepare_data(self):
        """Prépare les données avec validation temporelle"""
        print("\n📊 Préparation des données...")
        
        # Séparer features et target
        X = self.df.drop(columns=['gap_percentage'])
        y = self.df['gap_percentage']
        
        print(f"  • Features: {X.shape[1]}")
        print(f"  • Samples: {X.shape[0]}")
        
        return X, y
    
    def train_with_timeseries_cv(self, X, y):
        """Entraîne avec validation croisée temporelle"""
        print("\n🚀 ENTRAÎNEMENT AVEC VALIDATION TEMPORELLE")
        print("="*60)
        
        # Initialiser TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Modèles avec régularisation anti-overfitting
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=0.5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            ),
            'Ridge Regression': Ridge(alpha=10.0, random_state=42),
            'Linear Regression': LinearRegression(),
        }
        
        # Résultats par fold
        fold_results = {name: {'r2': [], 'rmse': [], 'mae': []} for name in self.models.keys()}
        
        print("  • Validation temporelle (3 folds):")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            print(f"\n    Fold {fold}:")
            print(f"      Train: {len(train_idx):,} samples")
            print(f"      Test:  {len(test_idx):,} samples")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Normalisation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            for name, model in self.models.items():
                # Entraînement
                model.fit(X_train_scaled, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test_scaled)
                
                # Métriques
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Stocker
                fold_results[name]['r2'].append(r2)
                fold_results[name]['rmse'].append(rmse)
                fold_results[name]['mae'].append(mae)
                
                print(f"      {name:20} R²={r2:+.3f}, RMSE={rmse:.3f}%")
        
        # Calculer les moyennes
        print("\n📊 RÉSULTATS MOYENS SUR 3 FOLDS:")
        print("-" * 40)
        
        final_results = {}
        for name in self.models.keys():
            avg_r2 = np.mean(fold_results[name]['r2'])
            avg_rmse = np.mean(fold_results[name]['rmse'])
            avg_mae = np.mean(fold_results[name]['mae'])
            
            final_results[name] = {
                'model': self.models[name],
                'avg_r2': avg_r2,
                'avg_rmse': avg_rmse,
                'avg_mae': avg_mae,
                'fold_r2': fold_results[name]['r2'],
                'fold_rmse': fold_results[name]['rmse']
            }
            
            print(f"  {name:20} R²={avg_r2:+.3f}, RMSE={avg_rmse:.3f}%, MAE={avg_mae:.3f}%")
        
        self.results = final_results
        return final_results
    
    def select_best_model(self):
        """Sélectionne le meilleur modèle"""
        print("\n🏆 SÉLECTION DU MEILLEUR MODÈLE")
        print("="*60)
        
        if not self.results:
            print("❌ Aucun résultat disponible")
            return None
        
        # Classer par RMSE moyen (plus bas = mieux)
        ranking = []
        for name, result in self.results.items():
            ranking.append({
                'Modèle': name,
                'RMSE Moyen': result['avg_rmse'],
                'R² Moyen': result['avg_r2'],
                'MAE Moyen': result['avg_mae'],
                'Stabilité R²': np.std(result['fold_r2'])  # Plus bas = plus stable
            })
        
        ranking_df = pd.DataFrame(ranking)
        ranking_df = ranking_df.sort_values('RMSE Moyen')
        
        print("\n📊 CLASSEMENT DES MODÈLES:")
        print(ranking_df.to_string(index=False))
        
        best_model_name = ranking_df.iloc[0]['Modèle']
        best_result = self.results[best_model_name]
        
        print(f"\n🎯 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   • RMSE Moyen: {best_result['avg_rmse']:.3f}%")
        print(f"   • R² Moyen: {best_result['avg_r2']:.3f}")
        print(f"   • MAE Moyen: {best_result['avg_mae']:.3f}%")
        
        # Interprétation
        r2 = best_result['avg_r2']
        if r2 > 0.6:
            print(f"   ✅ Excellente performance prédictive")
        elif r2 > 0.4:
            print(f"   ✅ Bonne performance réaliste")
        elif r2 > 0.2:
            print(f"   🟡 Performance acceptable")
        elif r2 > 0.0:
            print(f"   🔴 Performance faible mais positive")
        else:
            print(f"   🔴 Performance très faible")
        
        return best_model_name, ranking_df
    
    def save_model(self, best_model_name):
        """Sauvegarde le meilleur modèle"""
        print(f"\n💾 Sauvegarde: {best_model_name}")
        
        import os
        os.makedirs('../models', exist_ok=True)
        
        best_model = self.results[best_model_name]['model']
        
        # Package à sauvegarder
        model_package = {
            'model': best_model,
            'model_name': best_model_name,
            'metrics': {
                'avg_rmse': self.results[best_model_name]['avg_rmse'],
                'avg_r2': self.results[best_model_name]['avg_r2'],
                'avg_mae': self.results[best_model_name]['avg_mae']
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': f"ml_dataset_OPTIMIZED.csv - {len(self.df)} samples",
            'features_count': self.df.shape[1] - 1
        }
        
        # Sauvegarder
        model_path = '../models/best_cos_model_optimized.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        return model_path

def main():
    """Fonction principale"""
    print("="*70)
    print("🧠 ENTRAÎNEMENT ML OPTIMISÉ - AVEC GÉNÉRALISATION")
    print("="*70)
    
    trainer = OptimizedModelTrainer()
    
    try:
        # 1. Charger les données
        print("\n[1/4] 📥 CHARGEMENT")
        df = trainer.load_data()
        
        # 2. Préparer les données
        print("\n[2/4] 📊 PRÉPARATION")
        X, y = trainer.prepare_data()
        
        # 3. Entraîner avec validation temporelle
        print("\n[3/4] 🚀 ENTRAÎNEMENT")
        results = trainer.train_with_timeseries_cv(X, y)
        
        # 4. Sélectionner et sauvegarder
        print("\n[4/4] 🏆 SÉLECTION")
        best_model_name, ranking_df = trainer.select_best_model()
        
        if best_model_name:
            model_path = trainer.save_model(best_model_name)
            
            print("\n" + "="*70)
            print("✅ ENTRAÎNEMENT OPTIMISÉ TERMINÉ !")
            print("="*70)
            
            print(f"\n🎯 RÉSULTATS RÉALISTES:")
            print(f"  • Meilleur modèle: {best_model_name}")
            print(f"  • Précision (RMSE): {trainer.results[best_model_name]['avg_rmse']:.3f}%")
            print(f"  • Qualité (R²): {trainer.results[best_model_name]['avg_r2']:.3f}")
            print(f"  • Modèle sauvegardé: {model_path}")
            
            print(f"\n💡 INTERPRÉTATION:")
            r2 = trainer.results[best_model_name]['avg_r2']
            if r2 > 0.6:
                print("  • Le modèle explique bien la variation du COS")
                print("  • Utilisable pour des prédictions opérationnelles")
            elif r2 > 0.4:
                print("  • Le modèle a une capacité prédictive raisonnable")
                print("  • Utile pour identifier des tendances")
            elif r2 > 0.2:
                print("  • Le modèle capture une partie de la variabilité")
                print("  • À combiner avec une expertise métier")
            else:
                print("  • Le modèle a une capacité prédictive limitée")
                print("  • Considérer l'ajout de nouvelles données")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
