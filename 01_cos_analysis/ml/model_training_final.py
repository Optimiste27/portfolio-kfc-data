"""
ENTRAÎNEMENT ML FINAL - Version optimisée sans biais régionaux
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel

class FinalModelTrainer:
    def __init__(self, dataset_path='../data/processed/ml_dataset_OPTIMIZED.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.models = {}
        self.results = {}
        
    def load_and_preprocess(self):
        """Charge et prétraite les données pour réduire le biais régional"""
        print("📥 Chargement et prétraitement des données...")
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"✅ Dataset chargé: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        print(f"🎯 Target: gap_percentage (mean={self.df['gap_percentage'].mean():.2f}%, std={self.df['gap_percentage'].std():.2f}%)")
        
        # ANALYSE INITIALE
        print("\n🔍 Analyse des features dominantes...")
        
        # Calculer les corrélations avec la cible
        correlations = []
        feature_cols = [col for col in self.df.columns if col != 'gap_percentage']
        
        for col in feature_cols:
            try:
                corr = np.corrcoef(self.df[col], self.df['gap_percentage'])[0, 1]
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                continue
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("📊 Top 10 corrélations features-cible:")
        for i, (col, corr) in enumerate(correlations[:10]):
            print(f"  {i+1:2}. {col:30} : {corr:+.3f}")
        
        # Vérifier si une région domine trop
        region_cols = [col for col in feature_cols if col.startswith('region_')]
        if region_cols:
            print(f"\n🌍 {len(region_cols)} features de région détectées")
            for col in region_cols:
                if col in dict(correlations[:5]):  # Si dans top 5
                    print(f"  ⚠️  {col} est très corrélée avec la cible")
        
        # Séparer features et target
        X = self.df.drop(columns=['gap_percentage'])
        y = self.df['gap_percentage']
        
        return X, y
    
    def create_stratified_time_split(self, X, y, n_folds=3):
        """Crée des splits temporels stratifiés par région"""
        print("\n📊 Création de splits stratifiés...")
        
        # Identifier les restaurants par région
        region_cols = [col for col in X.columns if col.startswith('region_')]
        
        # Pour chaque restaurant, déterminer sa région principale
        if region_cols:
            # Créer une colonne de région (la région avec valeur 1)
            X_temp = X.copy()
            X_temp['main_region'] = X_temp[region_cols].idxmax(axis=1)
            
            # Compter les échantillons par région
            region_counts = X_temp['main_region'].value_counts()
            print(f"📈 Distribution par région:")
            for region, count in region_counts.items():
                percentage = count / len(X_temp) * 100
                print(f"  • {region:20} : {count:4} échantillons ({percentage:.1f}%)")
        
        # TimeSeriesSplit standard (nous ajusterons plus tard si nécessaire)
        tscv = TimeSeriesSplit(n_splits=n_folds)
        return tscv
    
    def train_optimized_models(self, X, y):
        """Entraîne des modèles optimisés avec régularisation"""
        print("\n🚀 ENTRAÎNEMENT DES MODÈLES OPTIMISÉS")
        print("="*60)
        
        # Créer le split temporel
        tscv = self.create_stratified_time_split(X, y, n_folds=3)
        
        # MODÈLES AVEC HYPERPARAMÈTRES OPTIMISÉS
        self.models = {
            'Random Forest Balanced': RandomForestRegressor(
                n_estimators=150,
                max_depth=6,  # Plus shallow pour éviter d'apprendre les régions
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.3,  # Utiliser peu de features pour forcer la généralisation
                max_samples=0.7,   # Bagging avec échantillonnage
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting Balanced': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,  # Learning rate plus bas
                max_depth=4,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.7,  # Stochastic gradient boosting
                max_features=0.5,
                random_state=42
            ),
            'Hist Gradient Boosting': HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=20,
                l2_regularization=1.0,  # Régularisation L2
                max_bins=64,
                random_state=42
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,  # Mix de L1 et L2
                random_state=42,
                max_iter=5000
            ),
            'Ridge Regression': Ridge(
                alpha=10.0,
                random_state=42
            )
        }
        
        # RÉSULTATS PAR FOLD
        fold_results = {name: {'r2': [], 'rmse': [], 'mae': []} for name in self.models.keys()}
        
        print("\n🔬 Validation temporelle (3 folds):")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            print(f"\n   Fold {fold}:")
            print(f"   {'='*40}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            print(f"   • Train: {len(X_train):,} samples")
            print(f"   • Test:  {len(X_test):,} samples")
            
            # Normalisation Robust (moins sensible aux outliers)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Sélection de features pour réduire la dominance des régions
            if fold == 1:  # Faire la sélection seulement sur le premier fold
                print(f"   • Sélection de features...")
                selector = SelectFromModel(
                    RandomForestRegressor(n_estimators=50, random_state=42),
                    threshold='median'  # Garder les 50% meilleures features
                )
                selector.fit(X_train_scaled, y_train)
                selected_features = selector.get_support()
                
                # Compter combien de features de région sont sélectionnées
                region_cols_idx = [i for i, col in enumerate(X.columns) 
                                 if col.startswith('region_')]
                selected_regions = sum(selected_features[i] for i in region_cols_idx)
                print(f"   • Features sélectionnées: {selected_features.sum()}/{len(selected_features)}")
                print(f"   • Régions sélectionnées: {selected_regions}/{len(region_cols_idx)}")
            
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
                
                # Afficher avec interprétation
                r2_str = f"{r2:+.3f}"
                if r2 > 0.5:
                    r2_str = f"✅{r2:+.3f}"
                elif r2 > 0.3:
                    r2_str = f"🟡{r2:+.3f}"
                elif r2 > 0.0:
                    r2_str = f"🔴{r2:+.3f}"
                
                print(f"   • {name:25} R²={r2_str}, RMSE={rmse:.3f}%")
        
        # CALCUL DES MOYENNES
        print("\n" + "="*60)
        print("📊 RÉSULTATS MOYENS SUR 3 FOLDS")
        print("="*60)
        
        final_results = {}
        for name in self.models.keys():
            avg_r2 = np.mean(fold_results[name]['r2'])
            avg_rmse = np.mean(fold_results[name]['rmse'])
            avg_mae = np.mean(fold_results[name]['mae'])
            std_r2 = np.std(fold_results[name]['r2'])
            
            final_results[name] = {
                'model': self.models[name],
                'avg_r2': avg_r2,
                'avg_rmse': avg_rmse,
                'avg_mae': avg_mae,
                'std_r2': std_r2,
                'fold_r2': fold_results[name]['r2'],
                'fold_rmse': fold_results[name]['rmse']
            }
            
            # Afficher avec indicateur de stabilité
            stability = "✅" if std_r2 < 0.2 else "⚠️ " if std_r2 < 0.3 else "🔴"
            
            print(f"\n   {name:25}")
            print(f"   {'─'*40}")
            print(f"   • R² Moyen:    {avg_r2:+.3f}")
            print(f"   • RMSE Moyen:  {avg_rmse:.3f}%")
            print(f"   • MAE Moyen:   {avg_mae:.3f}%")
            print(f"   • Stabilité:   {stability} (std R²: {std_r2:.3f})")
            
            # Valeurs par fold
            print(f"   • R² par fold: [{', '.join([f'{r:+.3f}' for r in fold_results[name]['r2']])}]")
        
        self.results = final_results
        return final_results
    
    def analyze_feature_importance(self, X, best_model_name):
        """Analyse l'importance des features du meilleur modèle"""
        print(f"\n🔍 ANALYSE DE L'IMPORTANCE DES FEATURES - {best_model_name}")
        print("="*60)
        
        best_model = self.results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Grouper par type de feature
            feature_categories = {
                'Régions': [col for col in X.columns if col.startswith('region_')],
                'Catégories': [col for col in X.columns if col.startswith('category_')],
                'Temporelles': ['month', 'day_of_week', 'is_weekend', 'week_of_year', 
                               'is_month_start', 'is_month_end'],
                'Prix/Ventes': ['selling_price', 'units_sold', 'qsp_score', 
                               'price_norm_by_category', 'price_to_units_ratio'],
                'Historiques': [col for col in X.columns if 'ma_' in col or 'std_' in col],
                'Tendances': [col for col in X.columns if 'trend' in col],
                'Interactions': ['rest_performance_30d', 'prod_popularity_in_rest']
            }
            
            # Calculer l'importance par catégorie
            category_importance = {}
            for cat_name, cat_features in feature_categories.items():
                cat_indices = [i for i, col in enumerate(X.columns) if col in cat_features]
                if cat_indices:
                    cat_importance = importances[cat_indices].sum()
                    category_importance[cat_name] = cat_importance
            
            # Afficher par catégorie
            print("\n📊 IMPORTANCE PAR CATÉGORIE DE FEATURE:")
            for cat_name, importance in sorted(category_importance.items(), 
                                             key=lambda x: x[1], reverse=True):
                percentage = importance * 100
                bar = "█" * int(percentage / 2)
                print(f"   • {cat_name:15} : {percentage:5.1f}% {bar}")
            
            # Top 15 features individuelles
            print(f"\n🏆 TOP 15 FEATURES INDIVIDUELLES:")
            for i, idx in enumerate(indices[:15]):
                feat_name = X.columns[idx]
                importance = importances[idx]
                percentage = importance * 100
                
                # Identifier le type
                feat_type = "Autre"
                for cat_name, cat_features in feature_categories.items():
                    if feat_name in cat_features:
                        feat_type = cat_name
                        break
                
                bar = "█" * int(percentage * 2)
                print(f"   {i+1:2}. {feat_name:30} : {percentage:5.1f}% {bar} ({feat_type})")
        
        else:
            print("   ⚠️  Ce modèle n'a pas d'importance de features disponible")
    
    def select_and_save_best_model(self, X):
        """Sélectionne et sauvegarde le meilleur modèle"""
        print("\n🏆 SÉLECTION DU MEILLEUR MODÈLE")
        print("="*60)
        
        if not self.results:
            print("❌ Aucun résultat disponible")
            return None
        
        # CRITÈRES DE SÉLECTION
        ranking = []
        for name, result in self.results.items():
            score = (
                result['avg_r2'] * 0.4 +          # Performance (40%)
                (1 - result['std_r2']) * 0.3 +    # Stabilité (30%)
                (1 - result['avg_rmse']/3) * 0.3  # Précision (30%)
            )
            
            ranking.append({
                'Modèle': name,
                'Score': score,
                'R² Moyen': result['avg_r2'],
                'RMSE Moyen': result['avg_rmse'],
                'Stabilité (std R²)': result['std_r2'],
                'R² par fold': result['fold_r2']
            })
        
        ranking_df = pd.DataFrame(ranking)
        ranking_df = ranking_df.sort_values('Score', ascending=False)
        
        print("\n📊 CLASSEMENT DES MODÈLES:")
        print(ranking_df[['Modèle', 'R² Moyen', 'RMSE Moyen', 'Stabilité (std R²)', 'Score']].to_string(index=False))
        
        best_model_name = ranking_df.iloc[0]['Modèle']
        best_result = self.results[best_model_name]
        
        print(f"\n🎯 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   • Score: {best_result['avg_r2']:.3f}")
        print(f"   • R² Moyen: {best_result['avg_r2']:.3f}")
        print(f"   • RMSE Moyen: {best_result['avg_rmse']:.3f}%")
        print(f"   • Stabilité: {best_result['std_r2']:.3f}")
        
        # INTERPRÉTATION
        r2 = best_result['avg_r2']
        stability = best_result['std_r2']
        
        print(f"\n💡 INTERPRÉTATION:")
        if r2 > 0.6 and stability < 0.15:
            print(f"   ✅ EXCELLENT! Modèle performant et stable")
            print(f"   🚀 Prêt pour le déploiement en production")
        elif r2 > 0.4 and stability < 0.25:
            print(f"   ✅ BON! Performance réaliste et assez stable")
            print(f"   📊 Utile pour l'analyse et la prise de décision")
        elif r2 > 0.2 and stability < 0.3:
            print(f"   🟡 ACCEPTABLE! Modèle avec limitations")
            print(f"   🔧 Peut être amélioré avec plus de données")
        else:
            print(f"   🔴 LIMITÉ! Performance faible ou instable")
            print(f"   💡 Considérer d'autres approches ou données")
        
        # ANALYSE DES FEATURES
        self.analyze_feature_importance(X, best_model_name)
        
        # SAUVEGARDE
        print(f"\n💾 Sauvegarde du modèle...")
        import os
        os.makedirs('../models', exist_ok=True)
        
        model_package = {
            'model': best_result['model'],
            'model_name': best_model_name,
            'metrics': {
                'avg_r2': best_result['avg_r2'],
                'avg_rmse': best_result['avg_rmse'],
                'avg_mae': best_result['avg_mae'],
                'std_r2': best_result['std_r2']
            },
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': f"ml_dataset_OPTIMIZED.csv - {len(self.df)} samples",
            'features_count': X.shape[1],
            'features_list': X.columns.tolist()
        }
        
        model_path = '../models/best_cos_model_final.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        
        # SAUVEGARDER UN RAPPORT
        report_path = '../reports/final_model_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("📋 RAPPORT DU MODÈLE FINAL - PRÉDICTION COS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🏆 Modèle sélectionné: {best_model_name}\n\n")
            
            f.write("📊 PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• R² Moyen: {best_result['avg_r2']:.3f}\n")
            f.write(f"• RMSE Moyen: {best_result['avg_rmse']:.3f}%\n")
            f.write(f"• MAE Moyen: {best_result['avg_mae']:.3f}%\n")
            f.write(f"• Stabilité (std R²): {best_result['std_r2']:.3f}\n\n")
            
            f.write("🎯 INTERPRÉTATION:\n")
            f.write("-" * 40 + "\n")
            if r2 > 0.6:
                f.write("• Performance excellente - Modèle très prédictif\n")
                f.write("• Peut être utilisé pour des prédictions opérationnelles\n")
            elif r2 > 0.4:
                f.write("• Bonne performance réaliste\n")
                f.write("• Utile pour identifier des tendances et patterns\n")
            elif r2 > 0.2:
                f.write("• Performance acceptable avec limitations\n")
                f.write("• À combiner avec une expertise métier\n")
            else:
                f.write("• Performance limitée - À améliorer\n")
                f.write("• Considérer l'ajout de nouvelles données\n")
            
            f.write(f"\n💾 Fichier modèle: {model_path}\n")
            f.write(f"📁 Dataset utilisé: ml_dataset_OPTIMIZED.csv\n")
        
        print(f"✅ Rapport généré: {report_path}")
        
        return best_model_name, model_path

def main():
    """Fonction principale"""
    print("="*70)
    print("🧠 ENTRAÎNEMENT ML FINAL - OPTIMISÉ SANS BIAIS RÉGIONAUX")
    print("="*70)
    
    trainer = FinalModelTrainer()
    
    try:
        # 1. Charger et analyser les données
        print("\n[1/4] 📊 ANALYSE DES DONNÉES")
        X, y = trainer.load_and_preprocess()
        
        # 2. Entraînement optimisé
        print("\n[2/4] 🚀 ENTRAÎNEMENT OPTIMISÉ")
        results = trainer.train_optimized_models(X, y)
        
        # 3. Sélection et sauvegarde
        print("\n[3/4] 🏆 SÉLECTION ET SAUVEGARDE")
        best_model_name, model_path = trainer.select_and_save_best_model(X)
        
        # 4. CONCLUSION
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT FINAL TERMINÉ AVEC SUCCÈS !")
        print("="*70)
        
        print(f"\n🎯 RÉSULTATS FINAUX:")
        best_result = trainer.results[best_model_name]
        print(f"  • Modèle: {best_model_name}")
        print(f"  • R²: {best_result['avg_r2']:.3f}")
        print(f"  • RMSE: {best_result['avg_rmse']:.3f}%")
        print(f"  • Stabilité: {best_result['std_r2']:.3f}")
        
        print(f"\n💾 ARTEFACTS CRÉÉS:")
        print(f"  • Modèle: ../models/best_cos_model_final.pkl")
        print(f"  • Rapport: ../reports/final_model_report.txt")
        print(f"  • Dataset: ../data/processed/ml_dataset_OPTIMIZED.csv")
        
        print(f"\n🚀 PRÊT POUR L'INTÉGRATION DASHBOARD!")
        print(f"   Le modèle peut maintenant être utilisé pour prédire le COS.")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()