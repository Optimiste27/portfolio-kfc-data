# ml/explore_features.py
"""
Exploration des données pour ML - Prédiction des écarts COS
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DataExplorer:
    def __init__(self, db_path='../src/cos_kfc.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Établit la connexion à la base"""
        self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def close(self):
        """Ferme la connexion"""
        if self.conn:
            self.conn.close()
    
    def load_ml_dataset(self):
        """Charge et prépare le dataset pour ML"""
        print("📊 CHARGEMENT DES DONNÉES POUR ML...")
        
        query = """
        SELECT 
            -- Identifiants
            t.transaction_id,
            t.date,
            
            -- Features temporelles
            strftime('%Y', t.date) as year,
            strftime('%m', t.date) as month,
            strftime('%d', t.date) as day,
            strftime('%w', t.date) as day_of_week,
            
            -- Features restaurant
            r.restaurant_id,
            r.restaurant_name,
            r.region,
            
            -- Features produit
            p.product_family_id,
            p.product_family_name,
            p.category as product_category,
            p.target_gap_percentage as product_target_gap,
            
            -- Features transaction
            t.theoretical_unit_cost,
            t.actual_unit_cost,
            t.selling_price,
            t.units_sold,
            t.waste_kg,
            t.qsp_score,
            t.operational_anomaly,
            t.revenue,
            
            -- Target variable
            t.gap_percentage,
            t.gap_severity
            
        FROM transactions t
        JOIN restaurants r ON t.restaurant_id = r.restaurant_id
        JOIN product_families p ON t.product_family_id = p.product_family_id
        ORDER BY t.date
        """
        
        df = pd.read_sql_query(query, self.connect())
        
        # Convertir les types
        df['date'] = pd.to_datetime(df['date'])
        df['operational_anomaly'] = df['operational_anomaly'].astype(int)
        
        print(f"✅ Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        return df
    
    def analyze_features(self, df):
        """Analyse statistique des features"""
        print("\n🔍 ANALYSE DES FEATURES:")
        print("="*60)
        
        # Types de données
        print("\n📐 TYPES DE DONNÉES:")
        print(df.dtypes.value_counts())
        
        # Statistiques descriptives
        print("\n📊 STATISTIQUES NUMÉRIQUES:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe().T
        print(stats[['count', 'mean', 'std', 'min', 'max']].round(2))
        
        # Variables catégorielles
        print("\n🏷️ VARIABLES CATÉGORIELLES:")
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col not in ['date', 'restaurant_name', 'product_family_name']:
                unique_vals = df[col].nunique()
                print(f"  {col:25}: {unique_vals:3} valeurs uniques")
        
        return numeric_cols, cat_cols
    
    def analyze_target(self, df):
        """Analyse de la variable cible (gap_percentage)"""
        print("\n🎯 ANALYSE DE LA VARIABLE CIBLE (gap_percentage):")
        print("="*60)
        
        target = df['gap_percentage']
        
        print(f"  • Moyenne: {target.mean():.2f}%")
        print(f"  • Médiane: {target.median():.2f}%")
        print(f"  • Std: {target.std():.2f}%")
        print(f"  • Min: {target.min():.2f}%")
        print(f"  • Max: {target.max():.2f}%")
        print(f"  • Skewness: {target.skew():.2f}")
        
        # Distribution par sévérité
        if 'gap_severity' in df.columns:
            print(f"\n📈 DISTRIBUTION PAR SÉVÉRITÉ:")
            severity_counts = df['gap_severity'].value_counts()
            for severity, count in severity_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  • {severity:15}: {count:5} ({percentage:.1f}%)")
        
        # Distribution temporelle
        print(f"\n📅 DISTRIBUTION TEMPORELLE:")
        monthly_avg = df.groupby(df['date'].dt.to_period('M'))['gap_percentage'].mean()
        print(f"  • Moyenne mensuelle: {monthly_avg.mean():.2f}%")
        print(f"  • Variabilité mensuelle: {monthly_avg.std():.2f}%")
    
    def correlation_analysis(self, df, numeric_cols):
        """Analyse des corrélations"""
        print("\n🔗 ANALYSE DES CORRÉLATIONS:")
        print("="*60)
        
        # Matrice de corrélation avec la target
        correlations = df[numeric_cols].corr()['gap_percentage'].sort_values(ascending=False)
        
        print("Corrélation avec gap_percentage:")
        for feature, corr in correlations.items():
            if feature != 'gap_percentage' and not pd.isna(corr):
                strength = "FORTE" if abs(corr) > 0.5 else "MOYENNE" if abs(corr) > 0.3 else "FAIBLE"
                direction = "positive" if corr > 0 else "negative"
                print(f"  • {feature:25}: {corr:6.3f} ({strength}, {direction})")
        
        return correlations
    
    def visualize_distributions(self, df):
        """Visualisations des distributions"""
        print("\n🎨 CRÉATION DES VISUALISATIONS...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribution des Variables Clés', fontsize=16, fontweight='bold')
        
        # 1. Distribution de gap_percentage
        axes[0, 0].hist(df['gap_percentage'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df['gap_percentage'].mean(), color='red', linestyle='--', label='Moyenne')
        axes[0, 0].axvline(4.0, color='green', linestyle='--', label='Cible (4%)')
        axes[0, 0].set_xlabel('Gap Percentage (%)')
        axes[0, 0].set_ylabel('Fréquence')
        axes[0, 0].set_title('Distribution des écarts COS')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gap par restaurant (top 10)
        top_restaurants = df.groupby('restaurant_name')['gap_percentage'].mean().nlargest(10)
        axes[0, 1].barh(range(len(top_restaurants)), top_restaurants.values)
        axes[0, 1].set_yticks(range(len(top_restaurants)))
        axes[0, 1].set_yticklabels(top_restaurants.index, fontsize=9)
        axes[0, 1].set_xlabel('Gap moyen (%)')
        axes[0, 1].set_title('Top 10 restaurants - Plus haut gap')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Gap par produit
        product_gap = df.groupby('product_category')['gap_percentage'].mean()
        axes[0, 2].bar(product_gap.index, product_gap.values)
        axes[0, 2].set_xlabel('Catégorie produit')
        axes[0, 2].set_ylabel('Gap moyen (%)')
        axes[0, 2].set_title('Gap par catégorie produit')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Évolution temporelle
        df['month'] = df['date'].dt.to_period('M').astype(str)
        monthly_gap = df.groupby('month')['gap_percentage'].mean()
        axes[1, 0].plot(monthly_gap.index, monthly_gap.values, marker='o', linewidth=2)
        axes[1, 0].axhline(y=4.0, color='green', linestyle='--', label='Cible 4%')
        axes[1, 0].set_xlabel('Mois')
        axes[1, 0].set_ylabel('Gap moyen (%)')
        axes[1, 0].set_title('Évolution mensuelle du gap')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Correlation heatmap (top 10 features)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 10:
            corr_with_target = numeric_df.corr()['gap_percentage'].abs().sort_values(ascending=False)
            top_features = corr_with_target[1:11].index  # Exclure la target elle-même
            corr_matrix = numeric_df[top_features].corr()
            
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 1].set_xticks(range(len(top_features)))
            axes[1, 1].set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features, fontsize=8)
            axes[1, 1].set_title('Matrice de corrélation (Top 10 features)')
            plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Anomalies vs gap
        if 'operational_anomaly' in df.columns:
            anomaly_groups = df.groupby('operational_anomaly')['gap_percentage'].agg(['mean', 'std', 'count'])
            x_pos = range(len(anomaly_groups))
            
            bars = axes[1, 2].bar(x_pos, anomaly_groups['mean'], 
                                  yerr=anomaly_groups['std'], 
                                  capsize=5, alpha=0.7, edgecolor='black')
            
            # Colorer les bars
            colors = ['green', 'red']
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels(['Normal', 'Anomalie'], fontsize=10)
            axes[1, 2].set_ylabel('Gap moyen (%)')
            axes[1, 2].set_title('Gap moyen avec/sans anomalies')
            axes[1, 2].grid(True, alpha=0.3, axis='y')
            
            # Ajouter les counts
            for i, count in enumerate(anomaly_groups['count']):
                axes[1, 2].text(i, anomaly_groups['mean'][i] + anomaly_groups['std'][i] + 0.1, 
                               f'n={count}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('../reports/ml_data_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualisations sauvegardées: reports/ml_data_analysis.png")
        
        return fig
    
    def generate_report(self, df):
        """Génère un rapport complet d'analyse"""
        print("\n" + "="*60)
        print("📋 RAPPORT D'ANALYSE ML - PRÉDICTION ÉCARTS COS")
        print("="*60)
        
        report = []
        report.append("="*60)
        report.append("📋 RAPPORT D'ANALYSE ML - PRÉDICTION ÉCARTS COS")
        report.append("="*60)
        report.append(f"\n📅 Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📊 Taille dataset: {len(df):,} observations")
        report.append(f"🎯 Variable cible: gap_percentage")
        
        # Qualité des données
        report.append("\n🔍 QUALITÉ DES DONNÉES:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_info = pd.DataFrame({
            'Valeurs manquantes': missing,
            'Pourcentage': missing_pct
        })
        missing_info = missing_info[missing_info['Valeurs manquantes'] > 0]
        
        if len(missing_info) > 0:
            report.append("Variables avec valeurs manquantes:")
            for idx, row in missing_info.iterrows():
                report.append(f"  • {idx:25}: {row['Valeurs manquantes']:4} ({row['Pourcentage']}%)")
        else:
            report.append("✅ Aucune valeur manquante détectée")
        
        # Insights pour ML
        report.append("\n💡 INSIGHTS POUR ML:")
        
        # 1. Variables prometteuses
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['gap_percentage'].abs().sort_values(ascending=False)
        top_correlations = correlations[1:6]  # Top 5 excluant la target
        
        report.append("\n1. Variables les plus corrélées avec gap_percentage:")
        for feature, corr in top_correlations.items():
            report.append(f"   • {feature:25}: {corr:.3f}")
        
        # 2. Recommandations feature engineering
        report.append("\n2. Recommandations de Feature Engineering:")
        report.append("   • Encodage one-hot: restaurant_name, product_category")
        report.append("   • Variables temporelles: jour_semaine, weekend, saison")
        report.append("   • Features lag: gap_percentage des jours précédents")
        report.append("   • Features rolling: moyenne mobile sur 7 jours")
        report.append("   • Interaction features: restaurant × produit")
        
        # 3. Stratégie de modélisation
        report.append("\n3. Stratégie de Modélisation Recommandée:")
        report.append("   • Problème: Régression (prédiction de pourcentage)")
        report.append("   • Métrique principale: RMSE (Root Mean Squared Error)")
        report.append("   • Métriques secondaires: MAE, R²")
        report.append("   • Validation: Time Series Split (garder ordre temporel)")
        report.append("   • Modèles à tester:")
        report.append("     1. Random Forest Regressor (baseline)")
        report.append("     2. Gradient Boosting (XGBoost/LightGBM)")
        report.append("     3. Time Series (LSTM/Prophet si pattern temporel fort)")
        
        # 4. Préparation des données
        report.append("\n4. Étapes de Préparation des Données:")
        report.append("   [ ] 1. Nettoyage valeurs aberrantes (outliers)")
        report.append("   [ ] 2. Feature engineering")
        report.append("   [ ] 3. Encodage variables catégorielles")
        report.append("   [ ] 4. Normalisation/standardisation")
        report.append("   [ ] 5. Split temporel (train/test)")
        report.append("   [ ] 6. Création features lag pour time series")
        
        # Enregistrer le rapport
        report_path = '../reports/ml_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Rapport généré: {report_path}")
        
        # Afficher le rapport
        print('\n'.join(report[:50]))  # Afficher les premières 50 lignes
        
        return report

def main():
    """Fonction principale"""
    print("="*70)
    print("🧠 EXPLORATION DONNÉES POUR ML - PRÉDICTION ÉCARTS COS")
    print("="*70)
    
    # Initialiser l'explorateur
    explorer = DataExplorer()
    
    try:
        # 1. Charger les données
        df = explorer.load_ml_dataset()
        
        # 2. Analyse des features
        numeric_cols, cat_cols = explorer.analyze_features(df)
        
        # 3. Analyse de la target
        explorer.analyze_target(df)
        
        # 4. Analyse des corrélations
        correlations = explorer.correlation_analysis(df, numeric_cols)
        
        # 5. Visualisations
        fig = explorer.visualize_distributions(df)
        
        # 6. Générer le rapport
        report = explorer.generate_report(df)
        
        print("\n" + "="*70)
        print("✅ EXPLORATION TERMINÉE - PRÊT POUR FEATURE ENGINEERING")
        print("="*70)
        
        # Afficher quelques insights clés
        print("\n🎯 INSIGHTS CLÉS:")
        print("• Variables numériques disponibles:", len(numeric_cols))
        print("• Variables catégorielles disponibles:", len(cat_cols))
        print("• Top 3 corrélations avec gap_percentage:")
        for i, (feature, corr) in enumerate(correlations.items()[:4]):
            if feature != 'gap_percentage':
                print(f"  {i}. {feature}: {corr:.3f}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        explorer.close()

if __name__ == "__main__":
    main()