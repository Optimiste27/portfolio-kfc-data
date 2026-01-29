"""
Crée un dataset ML réaliste sans overfitting
Version OPTIMISÉE - Pour modèles généralisables
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def create_optimized_ml_dataset():
    """Crée un dataset ML OPTIMISÉ pour la généralisation"""
    print("="*70)
    print("🔧 CRÉATION DATASET ML OPTIMISÉ - POUR GÉNÉRALISATION")
    print("="*70)
    
    # 1. Charger les données avec features additionnelles
    print("\n[1/4] 📥 CHARGEMENT DES DONNÉES ENRICHIES")
    print("-" * 40)
    
    db_path = '../src/cos_kfc.db'
    conn = sqlite3.connect(db_path)
    
    # Ajouter des informations contextuelles
    query = """
    SELECT 
        t.date,
        t.restaurant_id,
        r.region,
        t.product_family_id,
        p.category as product_category,
        t.selling_price,
        t.units_sold,
        t.qsp_score,
        t.gap_percentage
    FROM transactions t
    JOIN restaurants r ON t.restaurant_id = r.restaurant_id
    JOIN product_families p ON t.product_family_id = p.product_family_id
    ORDER BY t.date, t.restaurant_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convertir types
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"📊 Données brutes: {len(df):,} lignes")
    print(f"📅 Période: {df['date'].min().date()} au {df['date'].max().date()}")
    print(f"🏪 Restaurants: {df['restaurant_id'].nunique()}, Régions: {df['region'].nunique()}")
    print(f"📦 Produits: {df['product_family_id'].nunique()}, Catégories: {df['product_category'].nunique()}")
    print(f"🎯 Cible (gap_percentage): {df['gap_percentage'].mean():.2f}% ± {df['gap_percentage'].std():.2f}%")
    
    # 2. CRÉATION DE FEATURES OPTIMISÉES
    print("\n[2/4] 🛡️ CRÉATION DE FEATURES OPTIMISÉES")
    print("-" * 40)
    
    # Trier pour garantir l'ordre temporel
    df = df.sort_values(['restaurant_id', 'date']).reset_index(drop=True)
    
    # A. Features temporelles enrichies
    print("  • Features temporelles...")
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # B. Features agrégées par groupe (au lieu d'identifiants bruts)
    print("  • Features agrégées...")
    
    # Statistiques par restaurant (basées sur données passées)
    for window in [14, 30]:
        # Moyenne mobile des ventes
        df[f'rest_units_ma_{window}'] = df.groupby('restaurant_id')['units_sold'] \
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        
        # Écart-type des ventes (volatilité)
        df[f'rest_units_std_{window}'] = df.groupby('restaurant_id')['units_sold'] \
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())
        
        # Moyenne mobile QSP
        df[f'rest_qsp_ma_{window}'] = df.groupby('restaurant_id')['qsp_score'] \
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    
    # C. Features de prix normalisées
    print("  • Features de prix...")
    
    # Prix normalisé par catégorie
    df['price_norm_by_category'] = df.groupby('product_category')['selling_price'] \
        .transform(lambda x: (x - x.mean()) / x.std())
    
    # Ratio prix/ventes
    df['price_to_units_ratio'] = df['selling_price'] / (df['units_sold'] + 1)
    
    # D. Features d'interaction enrichies
    print("  • Features d'interaction...")
    
    # Performance relative du restaurant
    df['rest_performance_30d'] = df.groupby('restaurant_id')['units_sold'] \
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()) / \
        df.groupby('restaurant_id')['units_sold'] \
        .transform(lambda x: x.shift(31).rolling(60, min_periods=1).mean())
    
    # Popularité du produit dans le restaurant
    df['prod_popularity_in_rest'] = df.groupby(['restaurant_id', 'product_family_id'])['units_sold'] \
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()) / \
        df.groupby('restaurant_id')['units_sold'] \
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    
    # E. Encodage intelligent des catégories
    print("  • Encodage catégories...")
    
    # One-hot encoding des régions
    for region in df['region'].unique():
        col_name = f'region_{region}'.replace(' ', '_').replace('é', 'e')
        df[col_name] = (df['region'] == region).astype(int)
    
    # One-hot encoding des catégories produit
    for category in df['product_category'].unique():
        col_name = f'category_{category}'.replace(' ', '_').replace('/', '_')
        df[col_name] = (df['product_category'] == category).astype(int)
    
    # F. Features de tendance
    print("  • Features de tendance...")
    
    # Tendance des ventes sur 7 jours
    df['units_trend_7d'] = df.groupby('restaurant_id')['units_sold'] \
        .transform(lambda x: x.shift(1).rolling(7, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
        ))
    
    # Tendance QSP sur 14 jours
    df['qsp_trend_14d'] = df.groupby('restaurant_id')['qsp_score'] \
        .transform(lambda x: x.shift(1).rolling(14, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
        ))
    
    # 3. NETTOYAGE ET SÉLECTION DE FEATURES
    print("\n[3/4] 🧹 NETTOYAGE INTELLIGENT")
    print("-" * 40)
    
    # Liste des colonnes à CONSERVER (features informatives)
    features_to_keep = [
        # Temporelles
        'month', 'day_of_week', 'is_weekend', 'week_of_year',
        'is_month_start', 'is_month_end',
        
        # Ventes et prix
        'selling_price', 'units_sold', 'qsp_score',
        'price_norm_by_category', 'price_to_units_ratio',
        
        # Agrégations
        'rest_units_ma_14', 'rest_units_ma_30',
        'rest_units_std_14', 'rest_units_std_30',
        'rest_qsp_ma_14', 'rest_qsp_ma_30',
        
        # Interactions
        'rest_performance_30d', 'prod_popularity_in_rest',
        'units_trend_7d', 'qsp_trend_14d',
        
        # Cible
        'gap_percentage'
    ]
    
    # Ajouter les one-hot encodings
    features_to_keep.extend([col for col in df.columns if col.startswith('region_')])
    features_to_keep.extend([col for col in df.columns if col.startswith('category_')])
    
    # Garder seulement les colonnes importantes
    df_final = df[[col for col in features_to_keep if col in df.columns]].copy()
    
    # Supprimer les NaN
    initial_rows = len(df_final)
    df_final = df_final.dropna()
    rows_dropped = initial_rows - len(df_final)
    
    print(f"  • Lignes supprimées (NaN): {rows_dropped}")
    print(f"  • Dataset final: {len(df_final):,} lignes, {len(df_final.columns)} colonnes")
    
    # Statistiques
    feature_cols = [col for col in df_final.columns if col != 'gap_percentage']
    print(f"  • Features informatives: {len(feature_cols)}")
    print(f"  • Cible: gap_percentage ({df_final['gap_percentage'].mean():.2f}%)")
    
    # 4. VÉRIFICATION ET TEST
    print("\n[4/4] 🧪 TEST DE GÉNÉRALISATION")
    print("-" * 40)
    
    X = df_final[feature_cols]
    y = df_final['gap_percentage']
    
    # Test avec validation croisée temporelle
    try:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        
        print("  • Validation temporelle (3 folds)...")
        
        tscv = TimeSeriesSplit(n_splits=3)
        r2_scores = []
        rmse_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Normalisation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Modèle avec régularisation
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,  # Profondeur limitée pour éviter overfitting
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=0.5,  # Utiliser seulement 50% des features par arbre
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test_scaled)
            
            # Métriques
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            
            print(f"    Fold {fold}: R² = {r2:+.3f}, RMSE = {rmse:.3f}%")
        
        # Moyennes
        avg_r2 = np.mean(r2_scores)
        avg_rmse = np.mean(rmse_scores)
        
        print(f"\n📊 PERFORMANCE MOYENNE:")
        print(f"  • R²:  {avg_r2:+.3f}")
        print(f"  • RMSE: {avg_rmse:.3f}%")
        
        # Interprétation
        if avg_r2 > 0.6:
            print(f"  ✅ EXCELLENT! Modèle très prédictif")
        elif avg_r2 > 0.4:
            print(f"  ✅ BON! Performance réaliste")
        elif avg_r2 > 0.2:
            print(f"  🟡 ACCEPTABLE! Peut être amélioré")
        elif avg_r2 > 0.0:
            print(f"  🔴 FAIBLE! Modèle peu informatif")
        else:
            print(f"  🔴 TRÈS FAIBLE! Revoir les features")
        
        # Importance des features
        if len(r2_scores) > 0 and r2_scores[-1] > -1:  # Si dernier fold pas trop mauvais
            print(f"\n🏆 IMPORTANCE DES FEATURES (dernier fold):")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices, 1):
                if idx < len(feature_cols):
                    feat_name = feature_cols[idx]
                    importance = importances[idx]
                    print(f"  {i:2}. {feat_name:30} : {importance:.3f}")
        
    except Exception as e:
        print(f"  ⚠️  Test échoué: {str(e)[:100]}")
    
    # 5. SAUVEGARDE FINALE
    print("\n💾 SAUVEGARDE FINALE")
    print("-" * 40)
    
    os.makedirs('../data/processed', exist_ok=True)
    
    output_path = '../data/processed/ml_dataset_OPTIMIZED.csv'
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ Dataset optimisé sauvegardé: {output_path}")
    print(f"   • Taille: {len(df_final):,} lignes × {len(df_final.columns)} colonnes")
    print(f"   • Features: {len(feature_cols)}")
    print(f"   • Cible: gap_percentage")
    
    # Afficher les features
    print(f"\n📋 LISTE DES FEATURES ({len(feature_cols)}):")
    categories = {
        'Temporelles': ['month', 'day_of_week', 'is_weekend', 'week_of_year', 'is_month_start', 'is_month_end'],
        'Prix/Ventes': ['selling_price', 'units_sold', 'qsp_score', 'price_norm_by_category', 'price_to_units_ratio'],
        'Historiques': [col for col in feature_cols if 'ma_' in col or 'std_' in col],
        'Tendances': [col for col in feature_cols if 'trend' in col],
        'Interactions': ['rest_performance_30d', 'prod_popularity_in_rest'],
        'Régions': [col for col in feature_cols if col.startswith('region_')],
        'Catégories': [col for col in feature_cols if col.startswith('category_')]
    }
    
    for cat_name, cat_features in categories.items():
        if any(feat in feature_cols for feat in cat_features):
            actual_features = [f for f in cat_features if f in feature_cols]
            if actual_features:
                print(f"  • {cat_name}: {len(actual_features)} features")
    
    # 6. RECOMMANDATIONS FINALES
    print("\n" + "="*70)
    print("🎯 STRATÉGIE D'ENTRAÎNEMENT OPTIMISÉE")
    print("="*70)
    
    print("""
    📊 CE QUE VOUS DEVEZ FAIRE MAINTENANT:
    
    1. UTILISER CE DATASET OPTIMISÉ:
       • Fichier: ml_dataset_OPTIMIZED.csv
       • Features: Informatives et généralisables
       • Pas d'identifiants bruts qui causent l'overfitting
    
    2. MODIFIER model_training_clean.py:
       dataset_path = '../data/processed/ml_dataset_OPTIMIZED.csv'
    
    3. AJOUTER LA NORMALISATION dans model_training_clean.py:
       - StandardScaler pour toutes les features
       - TimeSeriesSplit pour validation
    
    4. HYPERPARAMÈTRES ANTI-OVERFITTING:
       • max_depth: 5-10
       • min_samples_split: 10-20  
       • max_features: 0.3-0.7
       • Utiliser Random Forest ou Gradient Boosting
    
    5. ATTENDRE DES RÉSULTATS RÉALISTES:
       • R²: 0.3 - 0.6 (C'EST BIEN !)
       • RMSE: 1.0% - 1.8%
       • Modèle qui généralise sur nouveaux restaurants
    """)
    
    print("\n🚀 COMMANDES À EXÉCUTER:")
    print("-" * 40)
    print("1. python create_proper_dataset.py  # Déjà fait")
    print("2. python fix_model_training.py     # Créer le fichier corrigé")
    print("3. python model_training_fixed.py   # Lancer l'entraînement optimisé")
    
    return df_final

def fix_model_training():
    """Crée une version corrigée de model_training_clean.py"""
    print("\n🔧 CRÉATION DU SCRIPT D'ENTRAÎNEMENT OPTIMISÉ")
    print("-" * 40)
    
    script_content = '''"""
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
        print("\\n📊 Préparation des données...")
        
        # Séparer features et target
        X = self.df.drop(columns=['gap_percentage'])
        y = self.df['gap_percentage']
        
        print(f"  • Features: {X.shape[1]}")
        print(f"  • Samples: {X.shape[0]}")
        
        return X, y
    
    def train_with_timeseries_cv(self, X, y):
        """Entraîne avec validation croisée temporelle"""
        print("\\n🚀 ENTRAÎNEMENT AVEC VALIDATION TEMPORELLE")
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
            print(f"\\n    Fold {fold}:")
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
        print("\\n📊 RÉSULTATS MOYENS SUR 3 FOLDS:")
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
        print("\\n🏆 SÉLECTION DU MEILLEUR MODÈLE")
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
        
        print("\\n📊 CLASSEMENT DES MODÈLES:")
        print(ranking_df.to_string(index=False))
        
        best_model_name = ranking_df.iloc[0]['Modèle']
        best_result = self.results[best_model_name]
        
        print(f"\\n🎯 MEILLEUR MODÈLE: {best_model_name}")
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
        print(f"\\n💾 Sauvegarde: {best_model_name}")
        
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
        print("\\n[1/4] 📥 CHARGEMENT")
        df = trainer.load_data()
        
        # 2. Préparer les données
        print("\\n[2/4] 📊 PRÉPARATION")
        X, y = trainer.prepare_data()
        
        # 3. Entraîner avec validation temporelle
        print("\\n[3/4] 🚀 ENTRAÎNEMENT")
        results = trainer.train_with_timeseries_cv(X, y)
        
        # 4. Sélectionner et sauvegarder
        print("\\n[4/4] 🏆 SÉLECTION")
        best_model_name, ranking_df = trainer.select_best_model()
        
        if best_model_name:
            model_path = trainer.save_model(best_model_name)
            
            print("\\n" + "="*70)
            print("✅ ENTRAÎNEMENT OPTIMISÉ TERMINÉ !")
            print("="*70)
            
            print(f"\\n🎯 RÉSULTATS RÉALISTES:")
            print(f"  • Meilleur modèle: {best_model_name}")
            print(f"  • Précision (RMSE): {trainer.results[best_model_name]['avg_rmse']:.3f}%")
            print(f"  • Qualité (R²): {trainer.results[best_model_name]['avg_r2']:.3f}")
            print(f"  • Modèle sauvegardé: {model_path}")
            
            print(f"\\n💡 INTERPRÉTATION:")
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
                print("  • Considérer l\'ajout de nouvelles données")
        
    except Exception as e:
        print(f"\\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    # Sauvegarder le script
    script_path = 'model_training_optimized.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ Script d'entraînement optimisé créé: {script_path}")
    print(f"   • Utilise ml_dataset_OPTIMIZED.csv")
    print(f"   • TimeSeriesSplit pour validation temporelle")
    print(f"   • Modèles avec régularisation anti-overfitting")
    
    return script_path

if __name__ == "__main__":
    # 1. Créer le dataset optimisé
    df_optimized = create_optimized_ml_dataset()
    
    # 2. Créer le script d'entraînement optimisé
    script_path = fix_model_training()
    
    print("\n" + "="*70)
    print("🚀 TOUT EST PRÊT POUR L'ENTRAÎNEMENT OPTIMISÉ !")
    print("="*70)
    
    print("\n📋 RÉCAPITULATIF:")
    print("-" * 40)
    print("✅ Dataset créé: ml_dataset_OPTIMIZED.csv")
    print("✅ Script créé: model_training_optimized.py")
    print("✅ Features: Informatives sans overfitting")
    print("✅ Validation: Temporelle (TimeSeriesSplit)")
    
    print("\n🎯 RÉSULTATS ATTENDUS:")
    print("-" * 40)
    print("• R²: 0.3 - 0.6 (réaliste et généralisable)")
    print("• RMSE: 1.0% - 1.8% (bonne précision pour COS)")
    print("• Pas d'overfitting (différence train/test < 0.2)")
    print("• Modèle utilisable sur nouveaux restaurants")
    
    print("\n⚡ COMMANDE À EXÉCUTER:")
    print("python model_training_optimized.py")