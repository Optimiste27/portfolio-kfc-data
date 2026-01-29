# ml/model_training_fixed.py
"""
Entraînement des modèles ML - Version corrigée
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
    print("⚠️  XGBoost non disponible")

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️  LightGBM non disponible")

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

class COSModelTrainerFixed:
    def __init__(self, dataset_path='../data/processed/ml_dataset.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Charge et prépare les données"""
        print("📥 Chargement et préparation des données...")
        self.df = pd.read_csv(self.dataset_path)
        
        # Identifier les colonnes non-numériques
        non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_cols) > 0:
            print(f"⚠️  Colonnes non-numériques détectées: {list(non_numeric_cols)}")
            
            # Essayer de convertir en numérique ou supprimer
            for col in non_numeric_cols:
                try:
                    # Essayer de convertir
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    print(f"  • Converti: {col}")
                except:
                    # Si échec, supprimer
                    print(f"  • Supprimé: {col}")
                    self.df = self.df.drop(columns=[col])
        
        # Supprimer les colonnes avec trop de NaN
        nan_threshold = 0.1  # 10% de NaN maximum
        nan_ratio = self.df.isnull().sum() / len(self.df)
        cols_to_drop = nan_ratio[nan_ratio > nan_threshold].index
        if len(cols_to_drop) > 0:
            print(f"⚠️  Colonnes avec >{nan_threshold:.0%} NaN supprimées: {list(cols_to_drop)}")
            self.df = self.df.drop(columns=cols_to_drop)
        
        # Remplir les NaN restants
        self.df = self.df.fillna(self.df.mean())
        
        print(f"✅ Dataset préparé: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        return self.df
    
    def prepare_train_test(self, test_size=0.2):
        """Prépare les sets d'entraînement et test"""
        print("\n📊 Préparation train/test...")
        
        # Séparer features et target
        X = self.df.drop(columns=['gap_percentage'], errors='ignore')
        y = self.df['gap_percentage']
        
        # Vérifier s'il y a une colonne date pour split temporel
        date_cols = [col for col in X.columns if 'date' in col.lower()]
        if date_cols and len(self.df) > 1000:
            # Utiliser la première colonne date trouvée
            date_col = date_cols[0]
            try:
                # Convertir en datetime si possible
                if not pd.api.types.is_datetime64_any_dtype(X[date_col]):
                    X[date_col] = pd.to_datetime(X[date_col], errors='coerce')
                
                # Split temporel (80/20)
                split_idx = int(len(X) * 0.8)
                X_train = X.iloc[:split_idx].drop(columns=[date_col], errors='ignore')
                X_test = X.iloc[split_idx:].drop(columns=[date_col], errors='ignore')
                y_train = y.iloc[:split_idx]
                y_test = y.iloc[split_idx:]
                
                print(f"  • Split temporel (80/20)")
                
            except:
                # Fallback: split aléatoire
                X_train, X_test, y_train, y_test = train_test_split(
                    X.drop(columns=date_cols, errors='ignore'), 
                    y, test_size=test_size, random_state=42
                )
                print(f"  • Split aléatoire (80/20)")
        else:
            # Split aléatoire
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            print(f"  • Split aléatoire (80/20)")
        
        print(f"  • Train: {len(X_train):,} échantillons")
        print(f"  • Test: {len(X_test):,} échantillons")
        print(f"  • Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Normalise les features"""
        print("⚖️  Normalisation des features...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convertir back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print("✅ Features normalisées")
        return X_train_scaled, X_test_scaled, scaler
    
    def initialize_models(self):
        """Initialise les modèles à tester"""
        print("\n🤖 Initialisation des modèles...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
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
        
        # Ajouter XGBoost si disponible
        if XGB_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        
        # Ajouter LightGBM si disponible
        if LGBM_AVAILABLE:
            self.models['LightGBM'] = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        
        print(f"✅ {len(self.models)} modèles initialisés")
        return self.models
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Entraîne tous les modèles"""
        print("\n🚀 Entraînement des modèles...")
        print("="*60)
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n🔧 {name}")
            
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
                
                print(f"  Train: MAE={train_mae:.3f}%, RMSE={train_rmse:.3f}%, R²={train_r2:.3f}")
                print(f"  Test:  MAE={test_mae:.3f}%, RMSE={test_rmse:.3f}%, R²={test_r2:.3f}")
                
                # Vérifier overfitting
                overfit = (test_rmse - train_rmse) / train_rmse
                if overfit > 0.2:
                    print(f"  ⚠️  Overfitting possible (+{overfit:.1%})")
                
            except Exception as e:
                print(f"  ❌ Erreur: {str(e)[:50]}")
                self.results[name] = None
        
        print("\n" + "="*60)
        print("✅ Entraînement terminé")
        
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
                    'Modèle': name,
                    'MAE Test': result['test_mae'],
                    'RMSE Test': result['test_rmse'],
                    'R² Test': result['test_r2'],
                    'R² Train': result['train_r2'],
                    'Overfit': (result['test_rmse'] - result['train_rmse']) / result['train_rmse']
                })
        
        if not results_df:
            print("❌ Aucun modèle entraîné avec succès")
            return None, None
        
        results_df = pd.DataFrame(results_df)
        results_df = results_df.sort_values('RMSE Test')
        
        # Afficher le classement
        print("\n🏆 CLASSEMENT DES MODÈLES:")
        print(results_df[['Modèle', 'RMSE Test', 'MAE Test', 'R² Test', 'Overfit']].to_string(index=False))
        
        # Meilleur modèle
        best_model_name = results_df.iloc[0]['Modèle']
        best_result = self.results[best_model_name]
        
        print(f"\n🎯 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   • RMSE: {best_result['test_rmse']:.3f}%")
        print(f"   • MAE: {best_result['test_mae']:.3f}%")
        print(f"   • R²: {best_result['test_r2']:.3f}")
        
        # Interprétation
        rmse = best_result['test_rmse']
        if rmse < 0.5:
            print(f"   ✅ Excellente précision (erreur < 0.5%)")
        elif rmse < 1.0:
            print(f"   🟡 Bonne précision (erreur < 1.0%)")
        elif rmse < 2.0:
            print(f"   🟠 Précision acceptable (erreur < 2.0%)")
        else:
            print(f"   🔴 Précision à améliorer")
        
        return results_df, best_model_name
    
    def analyze_feature_importance(self, X_train, best_model_name):
        """Analyse l'importance des features"""
        print(f"\n🔍 IMPORTANCE DES FEATURES - {best_model_name}")
        print("="*60)
        
        best_model = self.results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = X_train.columns
            
            # Créer un DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(20)
            
            print("\n🏆 Top 20 features les plus importantes:")
            for idx, row in importance_df.iterrows():
                print(f"  {row['Feature']:30}: {row['Importance']:.4f}")
            
            # Visualisation
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['Importance'][::-1])
            plt.yticks(range(len(importance_df)), importance_df['Feature'][::-1], fontsize=9)
            plt.xlabel('Importance')
            plt.title(f'Importance des Features - {best_model_name}')
            plt.tight_layout()
            
            # Sauvegarder
            import os
            os.makedirs('../reports', exist_ok=True)
            plt.savefig('../reports/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"\n✅ Graphique sauvegardé: reports/feature_importance.png")
            
            plt.show()
            
            return importance_df
        
        elif hasattr(best_model, 'coef_'):
            # Pour les modèles linéaires
            coefficients = best_model.coef_
            feature_names = X_train.columns
            
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            }).sort_values('Coefficient', key=abs, ascending=False).head(20)
            
            print("\n🏆 Top 20 coefficients (valeur absolue):")
            for idx, row in coef_df.iterrows():
                print(f"  {row['Feature']:30}: {row['Coefficient']:.4f}")
            
            return coef_df
        
        else:
            print("⚠️  Importance des features non disponible pour ce modèle")
            return None
    
    def visualize_results(self, y_test, best_model_name):
        """Visualise les résultats"""
        print("\n🎨 VISUALISATION DES RÉSULTATS")
        print("="*60)
        
        best_result = self.results[best_model_name]
        y_pred = best_result['y_pred_test']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Performance: {best_model_name}', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Valeurs Réelles (%)')
        axes[0, 0].set_ylabel('Prédictions (%)')
        axes[0, 0].set_title('Prédictions vs Réalité')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution des erreurs
        errors = y_pred - y_test
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0, color='red', linestyle='--', label='Erreur = 0')
        axes[0, 1].axvline(errors.mean(), color='green', linestyle='--', label=f'Moyenne: {errors.mean():.2f}')
        axes[0, 1].set_xlabel('Erreur de prédiction (%)')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].set_title(f'Distribution des erreurs\nMAE: {best_result["test_mae"]:.2f}%')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Comparaison des modèles (RMSE)
        models_list = []
        rmse_list = []
        
        for name, result in self.results.items():
            if result is not None:
                models_list.append(name)
                rmse_list.append(result['test_rmse'])
        
        # Trier
        sorted_pairs = sorted(zip(rmse_list, models_list))
        rmse_sorted, models_sorted = zip(*sorted_pairs)
        
        bars = axes[1, 0].barh(range(len(models_sorted)), rmse_sorted)
        axes[1, 0].set_yticks(range(len(models_sorted)))
        axes[1, 0].set_yticklabels(models_sorted)
        axes[1, 0].set_xlabel('RMSE (%)')
        axes[1, 0].set_title('Comparaison des modèles')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Colorer la meilleure barre
        bars[0].set_color('green')
        
        # 4. Prédictions vs Réalité (premiers 100 échantillons)
        sample_size = min(100, len(y_test))
        indices = np.arange(sample_size)
        
        axes[1, 1].plot(indices, y_test.iloc[:sample_size].values, 'b-', label='Réel', alpha=0.7, linewidth=2)
        axes[1, 1].plot(indices, y_pred[:sample_size], 'r--', label='Prédit', alpha=0.7, linewidth=2)
        axes[1, 1].fill_between(indices, 
                               y_test.iloc[:sample_size].values, 
                               y_pred[:sample_size],
                               alpha=0.2, color='gray')
        axes[1, 1].set_xlabel('Échantillon')
        axes[1, 1].set_ylabel('gap_percentage (%)')
        axes[1, 1].set_title(f'Prédictions vs Réalité (premiers {sample_size} échantillons)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        plt.savefig('../reports/model_performance.png', dpi=300, bbox_inches='tight')
        print("✅ Visualisations sauvegardées: reports/model_performance.png")
        
        plt.show()
        
        return fig
    
    def save_best_model(self, best_model_name, scaler=None):
        """Sauvegarde le meilleur modèle"""
        print(f"\n💾 SAUVEGARDE DU MODÈLE: {best_model_name}")
        
        best_model = self.results[best_model_name]['model']
        
        # Créer le dossier models
        import os
        os.makedirs('../models', exist_ok=True)
        
        # Sauvegarder le modèle
        model_path = f'../models/best_cos_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'model_name': best_model_name,
                'scaler': scaler,
                'metrics': {
                    'test_rmse': self.results[best_model_name]['test_rmse'],
                    'test_mae': self.results[best_model_name]['test_mae'],
                    'test_r2': self.results[best_model_name]['test_r2']
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        
        # Sauvegarder les métriques en JSON
        import json
        metrics = {
            'best_model': best_model_name,
            'test_rmse': float(self.results[best_model_name]['test_rmse']),
            'test_mae': float(self.results[best_model_name]['test_mae']),
            'test_r2': float(self.results[best_model_name]['test_r2']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metrics_path = '../models/model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Métriques sauvegardées: {metrics_path}")
        
        return model_path
    
    def generate_report(self, results_df, best_model_name):
        """Génère un rapport final"""
        print("\n📋 RAPPORT FINAL D'ENTRAÎNEMENT ML")
        print("="*60)
        
        report = []
        report.append("="*60)
        report.append("📋 RAPPORT ML - PRÉDICTION ÉCARTS COS")
        report.append("="*60)
        report.append(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📊 Dataset: {len(self.df)} échantillons, {len(self.df.columns)-1} features")
        
        report.append(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
        best_metrics = self.results[best_model_name]
        report.append(f"   • RMSE: {best_metrics['test_rmse']:.3f}%")
        report.append(f"   • MAE: {best_metrics['test_mae']:.3f}%")
        report.append(f"   • R²: {best_metrics['test_r2']:.3f}")
        
        report.append(f"\n📈 COMPARAISON DES MODÈLES:")
        report.append(results_df[['Modèle', 'RMSE Test', 'MAE Test', 'R² Test']].to_string(index=False))
        
        report.append(f"\n💡 RECOMMANDATIONS:")
        rmse = best_metrics['test_rmse']
        if rmse < 0.5:
            report.append("✅ Le modèle a une excellente précision et peut être utilisé pour:")
            report.append("   • Prédire les écarts COS futurs avec confiance")
            report.append("   • Identifier les risques d'anomalies")
            report.append("   • Optimiser les coûts opérationnels")
        elif rmse < 1.0:
            report.append("🟡 Le modèle a une bonne précision et peut être utilisé pour:")
            report.append("   • Prédire les tendances des écarts COS")
            report.append("   • Aider à la décision opérationnelle")
            report.append("   • Surveiller les performances")
        else:
            report.append("🟠 Le modèle a une précision acceptable. Suggestions d'amélioration:")
            report.append("   • Collecter plus de données")
            report.append("   • Améliorer le feature engineering")
            report.append("   • Essayer d'autres algorithmes")
        
        report.append(f"\n🚀 PROCHAINES ÉTAPES:")
        report.append("1. Intégrer le modèle au dashboard Streamlit")
        report.append("2. Créer une API de prédiction")
        report.append("3. Mettre en place un monitoring des prédictions")
        report.append("4. Re-entraîner périodiquement avec nouvelles données")
        
        # Sauvegarder le rapport
        report_path = '../reports/ml_training_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Rapport généré: {report_path}")
        
        # Afficher un résumé
        print('\n'.join(report[:25]))
        
        return report

def main():
    """Fonction principale"""
    print("="*70)
    print("🚀 ENTRAÎNEMENT MODÈLES ML - PRÉDICTION ÉCARTS COS")
    print("="*70)
    
    trainer = COSModelTrainerFixed()
    
    try:
        # 1. Charger et préparer les données
        print("\n[1/7] 📥 CHARGEMENT DES DONNÉES")
        df = trainer.load_and_prepare_data()
        
        # 2. Préparer train/test
        print("\n[2/7] 📊 PRÉPARATION TRAIN/TEST")
        X_train, X_test, y_train, y_test = trainer.prepare_train_test()
        
        # 3. Normaliser les features
        print("\n[3/7] ⚖️  NORMALISATION")
        X_train_scaled, X_test_scaled, scaler = trainer.scale_features(X_train, X_test)
        
        # 4. Initialiser les modèles
        print("\n[4/7] 🤖 INITIALISATION MODÈLES")
        trainer.initialize_models()
        
        # 5. Entraîner les modèles
        print("\n[5/7] 🚀 ENTRAÎNEMENT")
        results = trainer.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # 6. Évaluer les modèles
        print("\n[6/7] 📈 ÉVALUATION")
        results_df, best_model_name = trainer.evaluate_models()
        
        if results_df is None:
            print("❌ Échec de l'entraînement")
            return
        
        # 7. Analyser l'importance des features
        print("\n[7/7] 🔍 ANALYSE")
        importance_df = trainer.analyze_feature_importance(X_train_scaled, best_model_name)
        
        # Visualisations
        trainer.visualize_results(y_test, best_model_name)
        
        # Sauvegarder le modèle
        model_path = trainer.save_best_model(best_model_name, scaler)
        
        # Générer le rapport
        report = trainer.generate_report(results_df, best_model_name)
        
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT ML RÉUSSI !")
        print("="*70)
        print(f"\n🎯 RÉSULTATS:")
        print(f"  • Meilleur modèle: {best_model_name}")
        print(f"  • Précision (RMSE): {trainer.results[best_model_name]['test_rmse']:.3f}%")
        print(f"  • Modèle sauvegardé: {model_path}")
        print(f"\n🚀 Prêt pour l'intégration dans le dashboard!")
        
        return trainer, best_model_name
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    trainer, best_model_name = main()