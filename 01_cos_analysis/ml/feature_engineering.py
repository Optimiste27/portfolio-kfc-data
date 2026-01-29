# ml/feature_engineering.py
"""
Feature Engineering pour prédiction des écarts COS
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class COSFeatureEngineer:
    def __init__(self, db_path='../src/cos_kfc.db'):
        self.db_path = db_path
        self.conn = None
        
    def load_data(self):
        """Charge les données brutes depuis SQL"""
        print("📥 Chargement des données depuis SQL...")
        
        query = """
        SELECT 
            t.date,
            t.restaurant_id,
            r.restaurant_name,
            r.region,
            t.product_family_id,
            p.product_family_name,
            p.category as product_category,
            p.target_gap_percentage,
            t.theoretical_unit_cost,
            t.actual_unit_cost,
            t.selling_price,
            t.units_sold,
            t.waste_kg,
            t.qsp_score,
            t.operational_anomaly,
            t.revenue,
            t.gap_percentage,
            t.gap_severity
        FROM transactions t
        JOIN restaurants r ON t.restaurant_id = r.restaurant_id
        JOIN product_families p ON t.product_family_id = p.product_family_id
        ORDER BY t.date, t.restaurant_id
        """
        
        self.conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, self.conn)
        
        # Convertir types
        df['date'] = pd.to_datetime(df['date'])
        df['operational_anomaly'] = df['operational_anomaly'].astype(int)
        
        print(f"✅ Données chargées: {len(df)} lignes")
        return df
    
    def create_time_features(self, df):
        """Crée des features temporelles"""
        print("⏰ Création des features temporelles...")
        
        # Features basiques
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=lundi, 6=dimanche
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Features dérivées
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Saison (pour l'hémisphère nord)
        df['season'] = df['month'].apply(lambda m: 
            'Winter' if m in [12, 1, 2] else
            'Spring' if m in [3, 4, 5] else
            'Summer' if m in [6, 7, 8] else 'Autumn'
        )
        
        return df
    
    def create_restaurant_features(self, df):
        """Crée des features spécifiques aux restaurants"""
        print("🏪 Création des features restaurant...")
        
        # Calculer les statistiques historiques par restaurant
        restaurant_stats = df.groupby('restaurant_id').agg({
            'gap_percentage': ['mean', 'std', 'min', 'max'],
            'operational_anomaly': 'mean',
            'revenue': 'mean'
        }).round(3)
        
        restaurant_stats.columns = ['_'.join(col).strip() for col in restaurant_stats.columns.values]
        restaurant_stats = restaurant_stats.rename(columns={
            'gap_percentage_mean': 'rest_avg_gap',
            'gap_percentage_std': 'rest_std_gap',
            'gap_percentage_min': 'rest_min_gap',
            'gap_percentage_max': 'rest_max_gap',
            'operational_anomaly_mean': 'rest_anomaly_rate',
            'revenue_mean': 'rest_avg_revenue'
        })
        
        # Fusionner avec le dataframe principal
        df = df.merge(restaurant_stats, left_on='restaurant_id', right_index=True, how='left')
        
        return df
    
    def create_product_features(self, df):
        """Crée des features spécifiques aux produits"""
        print("🍗 Création des features produit...")
        
        # Statistiques par produit
        product_stats = df.groupby('product_family_id').agg({
            'gap_percentage': ['mean', 'std'],
            'units_sold': 'mean',
            'revenue': 'mean'
        }).round(3)
        
        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns.values]
        product_stats = product_stats.rename(columns={
            'gap_percentage_mean': 'prod_avg_gap',
            'gap_percentage_std': 'prod_std_gap',
            'units_sold_mean': 'prod_avg_units',
            'revenue_mean': 'prod_avg_revenue'
        })
        
        # Fusionner
        df = df.merge(product_stats, left_on='product_family_id', right_index=True, how='left')
        
        # Ratio prix/coût
        df['price_cost_ratio'] = df['selling_price'] / df['actual_unit_cost']
        df['theoretical_price_cost_ratio'] = df['selling_price'] / df['theoretical_unit_cost']
        df['cost_variance'] = df['actual_unit_cost'] - df['theoretical_unit_cost']
        df['cost_variance_pct'] = (df['cost_variance'] / df['theoretical_unit_cost']) * 100
        
        return df
    
    def create_lag_features(self, df, lags=[1, 2, 3, 7]):
        """Crée des features lag pour time series"""
        print("🔄 Création des features lag...")
        
        # Trier par date et restaurant
        df = df.sort_values(['restaurant_id', 'date'])
        
        # Features lag par restaurant
        for lag in lags:
            df[f'gap_lag_{lag}'] = df.groupby('restaurant_id')['gap_percentage'].shift(lag)
            df[f'anomaly_lag_{lag}'] = df.groupby('restaurant_id')['operational_anomaly'].shift(lag)
            df[f'revenue_lag_{lag}'] = df.groupby('restaurant_id')['revenue'].shift(lag)
        
        # Moyennes mobiles
        for window in [3, 7, 14]:
            df[f'gap_rolling_mean_{window}'] = df.groupby('restaurant_id')['gap_percentage'] \
                .rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'gap_rolling_std_{window}'] = df.groupby('restaurant_id')['gap_percentage'] \
                .rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
        
        return df
    
    def create_interaction_features(self, df):
        """Crée des features d'interaction"""
        print("🤝 Création des features d'interaction...")
        
        # Interactions restaurant × produit
        df['rest_prod_interaction'] = df['restaurant_id'].astype(str) + '_' + df['product_family_id'].astype(str)
        
        # Statistiques par combinaison restaurant-produit
        interaction_stats = df.groupby('rest_prod_interaction').agg({
            'gap_percentage': 'mean',
            'operational_anomaly': 'mean',
            'revenue': 'mean'
        }).round(3)
        
        interaction_stats.columns = [f'interaction_{col}' for col in interaction_stats.columns]
        
        # Fusionner
        df = df.merge(interaction_stats, left_on='rest_prod_interaction', right_index=True, how='left')
        
        # Features basées sur le ratio
        df['waste_per_unit'] = df['waste_kg'] / df['units_sold']
        df['revenue_per_unit'] = df['revenue'] / df['units_sold']
        df['profit_margin'] = (df['revenue'] - (df['actual_unit_cost'] * df['units_sold'])) / df['revenue']
        
        return df
    
    def create_advanced_features(self, df):
        """Crée des features avancées"""
        print("🚀 Création des features avancées...")
        
        # Distance à la cible
        df['gap_to_target'] = df['gap_percentage'] - df['target_gap_percentage']
        df['abs_gap_to_target'] = abs(df['gap_to_target'])
        
        # Performance relative
        df['gap_percentile'] = df.groupby(['restaurant_id', 'product_category'])['gap_percentage'] \
            .rank(pct=True)
        
        # Trend features
        df['gap_trend_3d'] = df['gap_percentage'] - df['gap_lag_3']
        df['gap_trend_7d'] = df['gap_percentage'] - df['gap_lag_7']
        
        # Flags
        df['high_waste_flag'] = (df['waste_kg'] > df['waste_kg'].quantile(0.75)).astype(int)
        df['low_qsp_flag'] = (df['qsp_score'] < df['qsp_score'].quantile(0.25)).astype(int)
        df['high_revenue_flag'] = (df['revenue'] > df['revenue'].quantile(0.75)).astype(int)
        
        return df
    
    def prepare_ml_dataset(self, df):
        """Prépare le dataset final pour ML"""
        print("🧹 Préparation du dataset ML...")
        
        # Supprimer les lignes avec NaN (causées par les lags)
        initial_rows = len(df)
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        print(f"  • Lignes supprimées (NaN): {rows_dropped}")
        
        # Séparer features et target
        target_cols = ['gap_percentage', 'gap_severity']
        
        # Identifier les colonnes à exclure
        exclude_cols = [
            'date', 'restaurant_name', 'product_family_name', 
            'rest_prod_interaction'
        ] + target_cols
        
        # Features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"  • Nombre de features: {len(feature_cols)}")
        print(f"  • Taille dataset final: {len(df)} lignes")
        
        return df, feature_cols, target_cols
    
    def get_feature_categories(self, df, feature_cols):
        """Catégorise les features pour le preprocessing"""
        numeric_features = []
        categorical_features = []
        
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                categorical_features.append(col)
        
        print(f"  • Features numériques: {len(numeric_features)}")
        print(f"  • Features catégorielles: {len(categorical_features)}")
        
        return numeric_features, categorical_features
    
    def create_preprocessor(self, numeric_features, categorical_features):
        """Crée le pipeline de preprocessing"""
        print("⚙️ Création du preprocessor...")
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def save_dataset(self, df, feature_cols, target_cols, filename='ml_dataset.csv'):
        """Sauvegarde le dataset préparé"""
        output_path = f'../data/processed/{filename}'
        
        # Sélectionner les colonnes importantes
        save_cols = feature_cols + target_cols + ['date', 'restaurant_name', 'product_family_name']
        df_to_save = df[save_cols].copy()
        
        df_to_save.to_csv(output_path, index=False)
        print(f"💾 Dataset sauvegardé: {output_path} ({len(df_to_save)} lignes)")
        
        return output_path
    
    def generate_feature_report(self, df, feature_cols):
        """Génère un rapport sur les features créées"""
        print("\n📋 RAPPORT DES FEATURES:")
        print("="*60)
        
        report = []
        report.append("="*60)
        report.append("📋 RAPPORT FEATURE ENGINEERING")
        report.append("="*60)
        report.append(f"\n📅 Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📊 Nombre total de features: {len(feature_cols)}")
        report.append(f"🎯 Variable cible: gap_percentage")
        
        # Catégories de features
        numeric_features = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        categorical_features = [col for col in feature_cols if col not in numeric_features]
        
        report.append(f"\n📈 Features numériques ({len(numeric_features)}):")
        for i, feat in enumerate(numeric_features[:15], 1):
            report.append(f"  {i:2}. {feat}")
        if len(numeric_features) > 15:
            report.append(f"  ... et {len(numeric_features)-15} autres")
        
        report.append(f"\n🏷️ Features catégorielles ({len(categorical_features)}):")
        for i, feat in enumerate(categorical_features[:10], 1):
            unique_vals = df[feat].nunique()
            report.append(f"  {i:2}. {feat:30} ({unique_vals} valeurs)")
        if len(categorical_features) > 10:
            report.append(f"  ... et {len(categorical_features)-10} autres")
        
        # Features les plus importantes (corrélation avec target)
        if 'gap_percentage' in df.columns:
            correlations = df[feature_cols + ['gap_percentage']].corr()['gap_percentage'].abs().sort_values(ascending=False)
            top_features = correlations[1:11]  # Exclure la target
            
            report.append(f"\n🔝 Top 10 features corrélées avec gap_percentage:")
            for i, (feat, corr) in enumerate(top_features.items(), 1):
                report.append(f"  {i:2}. {feat:30}: {corr:.3f}")
        
        # Informations sur les lags
        lag_features = [col for col in feature_cols if 'lag' in col or 'rolling' in col]
        if lag_features:
            report.append(f"\n🔄 Features temporelles (lags/rolling): {len(lag_features)}")
        
        # Sauvegarder le rapport
        report_path = '../reports/feature_engineering_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Rapport généré: {report_path}")
        
        # Afficher un résumé
        print('\n'.join(report[:30]))
        
        return report

def main():
    """Fonction principale de feature engineering"""
    print("="*70)
    print("🔧 FEATURE ENGINEERING POUR PRÉDICTION ÉCARTS COS")
    print("="*70)
    
    engineer = COSFeatureEngineer()
    
    try:
        # 1. Charger les données
        print("\n" + "="*70)
        print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
        print("="*70)
        df = engineer.load_data()
        
        # 2. Créer toutes les features
        print("\n" + "="*70)
        print("ÉTAPE 2: CRÉATION DES FEATURES")
        print("="*70)
        
        df = engineer.create_time_features(df)
        df = engineer.create_restaurant_features(df)
        df = engineer.create_product_features(df)
        df = engineer.create_lag_features(df, lags=[1, 2, 3, 7, 14])
        df = engineer.create_interaction_features(df)
        df = engineer.create_advanced_features(df)
        
        # 3. Préparer le dataset ML
        print("\n" + "="*70)
        print("ÉTAPE 3: PRÉPARATION DU DATASET ML")
        print("="*70)
        df, feature_cols, target_cols = engineer.prepare_ml_dataset(df)
        
        # 4. Catégoriser les features
        numeric_features, categorical_features = engineer.get_feature_categories(df, feature_cols)
        
        # 5. Créer le preprocessor
        preprocessor = engineer.create_preprocessor(numeric_features, categorical_features)
        
        # 6. Sauvegarder le dataset
        dataset_path = engineer.save_dataset(df, feature_cols, target_cols)
        
        # 7. Générer le rapport
        engineer.generate_feature_report(df, feature_cols)
        
        print("\n" + "="*70)
        print("✅ FEATURE ENGINEERING TERMINÉ AVEC SUCCÈS!")
        print("="*70)
        print(f"\n📊 RÉSULTATS FINAUX:")
        print(f"  • Dataset final: {len(df)} lignes")
        print(f"  • Nombre de features: {len(feature_cols)}")
        print(f"  • Features sauvegardées: {dataset_path}")
        print(f"  • Preprocessor créé: Prêt pour l'entraînement")
        
        return df, feature_cols, target_cols, preprocessor
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
    finally:
        if engineer.conn:
            engineer.conn.close()

if __name__ == "__main__":
    df, feature_cols, target_cols, preprocessor = main()