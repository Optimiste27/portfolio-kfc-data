# ml/model_training.py
"""
Entraînement des modèles ML pour prédiction des écarts COS
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
from lightgbm import LGBMRegressor

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

class COSModelTrainer:
    def __init__(self, dataset_path='../data/processed/ml_dataset.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Charge le dataset préparé"""
        print("📥 Chargement du dataset ML...")
        self.df = pd.read_csv(self.dataset_path)
        
        # Convertir date si présente
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"✅ Dataset chargé: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        
        return self.df
    
    def prepare_train_test(self, test_size=0.2, time_based=True):
        """Prépare les sets d'entraînement et test"""
        print("\n📊 Préparation train/test...")
        
        if time_based and 'date' in self.df.columns:
            # Split temporel (garder l'ordre chronologique)
            split_date = self.df['date'].quantile(0.8)
            train_mask = self.df['date'] < split_date
            test_mask = self.df['date'] >= split_date
            
            X_train = self.df[train_mask].drop(columns=['gap_percentage', 'gap_severity', 'date'], errors='ignore')
            y_train = self.df[train_mask]['gap_percentage']
            
            X_test = self.df[test_mask].drop(columns=['gap_percentage', 'gap_severity', 'date'], errors='ignore')
            y_test = self.df[test_mask]['gap_percentage']
            
            print(f"  • Split temporel (80/20)")
            print(f"  • Période train: {self.df[train_mask]['date'].min().date()} → {self.df[train_mask]['date'].max().date()}")
            print(f"  • Période test: {self.df[test_mask]['date'].min().date()} → {self.df[test_mask]['date'].max().date()}")
            
        else:
            # Split aléatoire
            X = self.df.drop(columns=['gap_percentage', 'gap_severity', 'date'], errors='ignore')
            y = self.df['gap_percentage']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            print(f"  • Split aléatoire (80/20)")
        
        print(f"  • Train: {len(X_train):,} échantillons")
        print(f"  • Test: {len(X_test):,} échantillons")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialise les modèles à tester"""
        print("\n🤖 Initialisation des modèles...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
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
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        }
        
        print(f"✅ {len(self.models)} modèles initialisés")
        return self.models
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Entraîne tous les modèles"""
        print("\n🚀 Entraînement des modèles...")
        print("="*60)
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n🔧 Entraînement: {name}")
            
            try:
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calcul des métriques
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Stocker les résultats
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
                
                print(f"  • Train MAE: {train_mae:.3f}%, RMSE: {train_rmse:.3f}%, R²: {train_r2:.3f}")
                print(f"  • Test  MAE: {test_mae:.3f}%, RMSE: {test_rmse:.3f}%, R²: {test_r2:.3f}")
                
                # Vérifier overfitting
                overfit_score = (test_rmse - train_rmse) / train_rmse
                if overfit_score > 0.2:
                    print(f"  ⚠️  Attention: Possible overfitting (diff: {overfit_score:.1%})")
                
            except Exception as e:
                print(f"  ❌ Erreur avec {name}: {str(e)[:50]}")
                self.results[name] = None
        
        print("\n" + "="*60)
        print("✅ Tous les modèles entraînés")
        
        return self.results
    
    def evaluate_models(self):
        """Évalue et compare les modèles"""
        print("\n📈 ÉVALUATION DES MODÈLES")
        print("="*60)
        
        # Créer un DataFrame avec les résultats
        results_df = []
        
        for name, result in self.results.items():
            if result is not None:
                results_df.append({
                    'Model': name,
                    'Test MAE': result['test_mae'],
                    'Test RMSE': result['test_rmse'],
                    'Test R²': result['test_r2'],
                    'Train R²': result['train_r2'],
                    'Overfit Score': (result['test_rmse'] - result['train_rmse']) / result['train_rmse']
                })
        
        results_df = pd.DataFrame(results_df)
        results_df = results_df.sort_values('Test RMSE')
        
        # Afficher le classement
        print("\n🏆 CLASSEMENT DES MODÈLES (par RMSE):")
        print(results_df[['Model', 'Test RMSE', 'Test MAE', 'Test R²', 'Overfit Score']].to_string(index=False))
        
        # Meilleur modèle
        best_model_name = results_df.iloc[0]['Model']
        best_result = self.results[best_model_name]
        
        print(f"\n🎯 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   • Test RMSE: {best_result['test_rmse']:.3f}%")
        print(f"   • Test MAE: {best_result['test_mae']:.3f}%")
        print(f"   • Test R²: {best_result['test_r2']:.3f}")
        
        return results_df, best_model_name
    
    def feature_importance_analysis(self, X_train):
        """Analyse l'importance des features du meilleur modèle"""
        print("\n🔍 ANALYSE IMPORTANCE DES FEATURES")
        print("="*60)
        
        # Trouver le meilleur modèle
        results_df, best_model_name = self.evaluate_models()
        best_model = self.results[best_model_name]['model']
        
        # Vérifier si le modèle a feature_importances_
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = X_train.columns
            
            # Créer un DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(20)
            
            print(f"\n🏆 Top 20 features les plus importantes ({best_model_name}):")
            for idx, row in importance_df.iterrows():
                print(f"  {row['feature']:30}: {row['importance']:.4f}")
            
            # Visualisation
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'][::-1])
            plt.yticks(range(len(importance_df)), importance_df['feature'][::-1])
            plt.xlabel('Importance')
            plt.title(f'Importance des Features - {best_model_name}')
            plt.tight_layout()
            plt.savefig('../reports/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"✅ Graphique sauvegardé: reports/feature_importance.png")
            
            return importance_df
        
        elif hasattr(best_model, 'coef_'):
            # Pour les modèles linéaires
            coefficients = best_model.coef_
            feature_names = X_train.columns
            
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients
            }).sort_values('coefficient', key=abs, ascending=False).head(20)
            
            print(f"\n🏆 Top 20 coefficients (valeur absolue) ({best_model_name}):")
            for idx, row in coef_df.iterrows():
                print(f"  {row['feature']:30}: {row['coefficient']:.4f}")
            
            return coef_df
        
        else:
            print(f"⚠️  {best_model_name} n'a pas d'importance de features disponible")
            return None
    
    def visualize_predictions(self, y_test, best_model_name):
        """Visualise les prédictions vs réalité"""
        print("\n🎨 VISUALISATION DES PRÉDICTIONS")
        print("="*60)
        
        best_result = self.results[best_model_name]
        y_pred = best_result['y_pred_test']
        
        # Créer les visualisations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Performance du Modèle: {best_model_name}', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot prédictions vs réalité
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Valeurs Réelles (gap_percentage)')
        axes[0, 0].set_ylabel('Prédictions')
        axes[0, 0].set_title('Prédictions vs Réalité')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution des erreurs
        errors = y_pred - y_test
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Erreur de prédiction')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].set_title(f'Distribution des erreurs (MAE: {best_result["test_mae"]:.2f}%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Comparaison des modèles (RMSE)
        models_list = []
        rmse_list = []
        
        for name, result in self.results.items():
            if result is not None:
                models_list.append(name)
                rmse_list.append(result['test_rmse'])
        
        sorted_idx = np.argsort(rmse_list)
        models_sorted = [models_list[i] for i in sorted_idx]
        rmse_sorted = [rmse_list[i] for i in sorted_idx]
        
        axes[1, 0].barh(range(len(models_sorted)), rmse_sorted)
        axes[1, 0].set_yticks(range(len(models_sorted)))
        axes[1, 0].set_yticklabels(models_sorted)
        axes[1, 0].set_xlabel('RMSE (%)')
        axes[1, 0].set_title('Comparaison des modèles (RMSE plus bas = mieux)')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Time series des prédictions (si date disponible)
        if hasattr(self, 'df') and 'date' in self.df.columns:
            # Prendre un échantillon pour la clarté
            sample_idx = np.random.choice(len(y_test), size=min(100, len(y_test)), replace=False)
            sample_dates = self.df.loc[y_test.index[sample_idx], 'date'] if 'date' in self.df.columns else None
            
            if sample_dates is not None:
                axes[1, 1].scatter(sample_dates, y_test.iloc[sample_idx], alpha=0.7, label='Réel', s=20)
                axes[1, 1].scatter(sample_dates, y_pred[sample_idx], alpha=0.7, label='Prédit', s=20)
                axes[1, 1].set_xlabel('Date')
                axes[1, 1].set_ylabel('gap_percentage (%)')
                axes[1, 1].set_title('Prédictions vs Réalité (échantillon)')
                axes[1, 1].legend()
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../reports/model_predictions.png', dpi=300, bbox_inches='tight')
        print(f"✅ Visualisations sauvegardées: reports/model_predictions.png")
        
        return fig
    
    def save_best_model(self, best_model_name, filename='best_cos_model.pkl'):
        """Sauvegarde le meilleur modèle"""
        print(f"\n💾 Sauvegarde du meilleur modèle: {best_model_name}")
        
        best_model = self.results[best_model_name]['model']
        model_path = f'../models/{filename}'
        
        # Créer le dossier models s'il n'existe pas
        import os
        os.makedirs('../models', exist_ok=True)
        
        # Sauvegarder le modèle
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        
        # Sauvegarder les métriques
        metrics = {
            'model_name': best_model_name,
            'test_mae': self.results[best_model_name]['test_mae'],
            'test_rmse': self.results[best_model_name]['test_rmse'],
            'test_r2': self.results[best_model_name]['test_r2'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        metrics_path = f'../models/{filename.replace(".pkl", "_metrics.json")}'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Métriques sauvegardées: {metrics_path}")
        
        return model_path
    
    def generate_training_report(self, results_df, best_model_name):
        """Génère un rapport d'entraînement"""
        print("\n📋 RAPPORT D'ENTRAÎNEMENT ML")
        print("="*60)
        
        report = []
        report.append("="*60)
        report.append("📋 RAPPORT D'ENTRAÎNEMENT ML - PRÉDICTION ÉCARTS COS")
        report.append("="*60)
        report.append(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📊 Taille dataset: {len(self.df):,} échantillons")
        report.append(f"🎯 Variable cible: gap_percentage")
        
        report.append(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
        best_metrics = self.results[best_model_name]
        report.append(f"   • Test RMSE: {best_metrics['test_rmse']:.3f}%")
        report.append(f"   • Test MAE: {best_metrics['test_mae']:.3f}%")
        report.append(f"   • Test R²: {best_metrics['test_r2']:.3f}")
        
        report.append(f"\n📈 PERFORMANCE TOUS MODÈLES:")
        report.append(results_df[['Model', 'Test RMSE', 'Test MAE', 'Test R²']].to_string(index=False))
        
        # Interprétation des résultats
        report.append(f"\n💡 INTERPRÉTATION:")
        rmse = best_metrics['test_rmse']
        if rmse < 0.5:
            report.append(f"✅ Excellente précision (erreur < 0.5%)")
        elif rmse < 1.0:
            report.append(f"🟡 Bonne précision (erreur < 1.0%)")
        elif rmse < 2.0:
            report.append(f"🟠 Précision acceptable (erreur < 2.0%)")
        else:
            report.append(f"🔴 Précision à améliorer (erreur > 2.0%)")
        
        report.append(f"\n🚀 RECOMMANDATIONS:")
        report.append("1. Utiliser ce modèle pour prédire les écarts COS futurs")
        report.append("2. Intégrer au dashboard Streamlit pour visualisation")
        report.append("3. Mettre en place un monitoring des prédictions")
        report.append("4. Re-entraîner périodiquement avec nouvelles données")
        
        # Sauvegarder le rapport
        report_path = '../reports/model_training_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Rapport généré: {report_path}")
        
        # Afficher un résumé
        print('\n'.join(report[:25]))
        
        return report

def main():
    """Fonction principale d'entraînement"""
    print("="*70)
    print("🚀 ENTRAÎNEMENT MODÈLES ML - PRÉDICTION ÉCARTS COS")
    print("="*70)
    
    trainer = COSModelTrainer()
    
    try:
        # 1. Charger les données
        print("\n" + "="*70)
        print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
        print("="*70)
        df = trainer.load_data()
        
        # 2. Préparer train/test
        print("\n" + "="*70)
        print("ÉTAPE 2: PRÉPARATION TRAIN/TEST")
        print("="*70)
        X_train, X_test, y_train, y_test = trainer.prepare_train_test(time_based=True)
        
        # 3. Initialiser les modèles
        print("\n" + "="*70)
        print("ÉTAPE 3: INITIALISATION MODÈLES")
        print("="*70)
        trainer.initialize_models()
        
        # 4. Entraîner les modèles
        print("\n" + "="*70)
        print("ÉTAPE 4: ENTRAÎNEMENT DES MODÈLES")
        print("="*70)
        results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        # 5. Évaluer les modèles
        print("\n" + "="*70)
        print("ÉTAPE 5: ÉVALUATION DES MODÈLES")
        print("="*70)
        results_df, best_model_name = trainer.evaluate_models()
        
        # 6. Analyser l'importance des features
        print("\n" + "="*70)
        print("ÉTAPE 6: ANALYSE IMPORTANCE FEATURES")
        print("="*70)
        importance_df = trainer.feature_importance_analysis(X_train)
        
        # 7. Visualiser les prédictions
        print("\n" + "="*70)
        print("ÉTAPE 7: VISUALISATION DES PRÉDICTIONS")
        print("="*70)
        fig = trainer.visualize_predictions(y_test, best_model_name)
        
        # 8. Sauvegarder le meilleur modèle
        print("\n" + "="*70)
        print("ÉTAPE 8: SAUVEGARDE DU MODÈLE")
        print("="*70)
        model_path = trainer.save_best_model(best_model_name)
        
        # 9. Générer le rapport
        print("\n" + "="*70)
        print("ÉTAPE 9: RAPPORT FINAL")
        print("="*70)
        report = trainer.generate_training_report(results_df, best_model_name)
        
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT ML TERMINÉ AVEC SUCCÈS!")
        print("="*70)
        print(f"\n🎯 RÉSULTATS CLÉS:")
        print(f"  • Meilleur modèle: {best_model_name}")
        print(f"  • Précision (RMSE): {trainer.results[best_model_name]['test_rmse']:.3f}%")
        print(f"  • Modèle sauvegardé: {model_path}")
        print(f"  • Prêt pour l'intégration dans le dashboard!")
        
        return trainer, best_model_name
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    trainer, best_model_name = main()