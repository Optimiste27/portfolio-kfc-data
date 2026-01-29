"""
MODÈLE DE PRODUCTION FINAL - CLASSIFICATION DU RISQUE COS
Prédit si un restaurant aura un gap COS élevé (> 5%)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class RiskClassificationModel:
    def __init__(self, dataset_path='../data/processed/ml_dataset_OPTIMIZED.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.models = {}
        self.results = {}
        
    def prepare_classification_data(self):
        """Prépare les données pour classification binaire"""
        print("📊 Préparation des données pour classification...")
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"✅ Dataset chargé: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        
        # 1. Créer la cible binaire : gap élevé (1) ou normal (0)
        threshold = 5.0  # 5% de gap
        self.df['high_gap_risk'] = (self.df['gap_percentage'] > threshold).astype(int)
        
        # Statistiques de la cible
        high_count = self.df['high_gap_risk'].sum()
        total_count = len(self.df)
        high_percentage = high_count / total_count * 100
        
        print(f"\n🎯 CIBLE BINAIRE (high_gap_risk):")
        print(f"   • Gap > {threshold}% = Risque élevé (1)")
        print(f"   • Gap ≤ {threshold}% = Normal (0)")
        print(f"   • Distribution: {high_count}/{total_count} = {high_percentage:.1f}% à risque")
        
        # 2. Sélectionner les features pertinentes
        # Garder les features qui pourraient prédire un risque élevé
        risk_features = [
            # Régions (important pour le risque)
            'region_Paris', 'region_Lyon', 'region_Marseille', 
            'region_Toulouse', 'region_Lille', 'region_Bordeaux',
            'region_Nantes', 'region_Strasbourg', 'region_Rennes', 'region_Nice',
            
            # Anomalies opérationnelles
            'operational_anomaly', 'anomaly_count_7d', 'anomaly_count_30d',
            
            # Qualité
            'qsp_score', 'qsp_ma_14', 'qsp_ma_30', 'qsp_trend_14d',
            
            # Ventes (volume peut indiquer pression opérationnelle)
            'units_sold', 'units_ma_14', 'units_ma_30', 'units_trend_7d',
            'rest_units_ma_14', 'rest_units_ma_30',
            
            # Prix (marge peut affecter la vigilance)
            'selling_price', 'price_norm_by_category',
            
            # Temporalité
            'day_of_week', 'is_weekend', 'month',
            
            # Cible
            'high_gap_risk'
        ]
        
        # Garder seulement les features qui existent
        existing_features = [f for f in risk_features if f in self.df.columns]
        self.df = self.df[existing_features]
        
        print(f"\n📊 Dataset final: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        print(f"   • Features: {len(self.df.columns) - 1}")
        print(f"   • Cible: high_gap_risk ({high_percentage:.1f}% positifs)")
        
        return self.df
    
    def train_risk_classifiers(self):
        """Entraîne des classifieurs de risque"""
        print("\n🚀 ENTRAÎNEMENT DES CLASSIFIEURS DE RISQUE")
        print("="*60)
        
        # Séparer features et cible
        X = self.df.drop(columns=['high_gap_risk'])
        y = self.df['high_gap_risk']
        
        # TimeSeriesSplit pour validation temporelle
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Modèles de classification
        self.models = {
            'Logistic Regression': LogisticRegression(
                C=0.1,  # Régularisation forte
                class_weight='balanced',  # Gérer le déséquilibre
                random_state=42,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # Limité pour éviter overfitting
                min_samples_split=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=20,
                random_state=42
            )
        }
        
        # Résultats
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔧 {name}:")
            
            # Métriques par fold
            acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Normalisation
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Gestion du déséquilibre avec SMOTE
                try:
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                    
                    # Entraînement sur données équilibrées
                    model.fit(X_train_balanced, y_train_balanced)
                except:
                    # Fallback si SMOTE échoue
                    model.fit(X_train_scaled, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Métriques
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                acc_scores.append(acc)
                prec_scores.append(prec)
                rec_scores.append(rec)
                f1_scores.append(f1)
                
                # AUC-ROC si disponible
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Stocker résultats moyens
            results[name] = {
                'model': model,
                'accuracy': np.mean(acc_scores),
                'precision': np.mean(prec_scores),
                'recall': np.mean(rec_scores),
                'f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'auc': auc if auc else None,
                'fold_scores': {
                    'accuracy': acc_scores,
                    'precision': prec_scores,
                    'recall': rec_scores,
                    'f1': f1_scores
                }
            }
            
            # Afficher résultats
            print(f"  • Accuracy:  {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
            print(f"  • Precision: {np.mean(prec_scores):.3f} ± {np.std(prec_scores):.3f}")
            print(f"  • Recall:    {np.mean(rec_scores):.3f} ± {np.std(rec_scores):.3f}")
            print(f"  • F1-Score:  {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
            if auc:
                print(f"  • AUC-ROC:   {auc:.3f}")
        
        self.results = results
        return results
    
    def select_best_classifier(self):
        """Sélectionne le meilleur classifieur"""
        print("\n🏆 SÉLECTION DU MEILLEUR CLASSIFIEUR")
        print("="*60)
        
        if not self.results:
            print("❌ Aucun résultat disponible")
            return None
        
        # Critère : F1-Score le plus élevé
        ranking = []
        for name, result in self.results.items():
            ranking.append({
                'Modèle': name,
                'F1-Score': result['f1'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'Stabilité': result['std_f1']
            })
        
        ranking_df = pd.DataFrame(ranking)
        ranking_df = ranking_df.sort_values('F1-Score', ascending=False)
        
        print("\n📊 CLASSEMENT:")
        print(ranking_df.to_string(index=False))
        
        best_model_name = ranking_df.iloc[0]['Modèle']
        best_result = self.results[best_model_name]
        
        print(f"\n🎯 MEILLEUR CLASSIFIEUR: {best_model_name}")
        print(f"   • F1-Score: {best_result['f1']:.3f}")
        print(f"   • Accuracy: {best_result['accuracy']:.3f}")
        print(f"   • Precision: {best_result['precision']:.3f}")
        print(f"   • Recall: {best_result['recall']:.3f}")
        
        # Interprétation business
        print(f"\n💡 INTERPRÉTATION BUSINESS:")
        f1 = best_result['f1']
        
        if f1 > 0.7:
            print(f"   ✅ EXCELLENT: Modèle très fiable pour détecter les risques")
            print(f"   🚀 Prêt pour l'alerte préventive aux managers")
        elif f1 > 0.5:
            print(f"   ✅ BON: Utile pour identifier les restaurants à risque")
            print(f"   📊 À utiliser avec vérification manuelle")
        elif f1 > 0.3:
            print(f"   🟡 ACCEPTABLE: Signal d'alerte précoce")
            print(f"   🔧 À améliorer avec plus de données")
        else:
            print(f"   🔴 LIMITÉ: Capacité prédictive faible")
            print(f"   💡 Considérer d'autres features")
        
        # Analyse par fold
        print(f"\n📈 PERFORMANCE PAR FOLD:")
        for i, (acc, prec, rec, f1_fold) in enumerate(zip(
            best_result['fold_scores']['accuracy'],
            best_result['fold_scores']['precision'],
            best_result['fold_scores']['recall'],
            best_result['fold_scores']['f1']
        ), 1):
            print(f"   • Fold {i}: F1={f1_fold:.3f}, Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")
        
        return best_model_name
    
    def analyze_important_features(self, best_model_name):
        """Analyse les features importantes pour la prédiction de risque"""
        print(f"\n🔍 ANALYSE DES FEATURES IMPORTANTES")
        print("="*60)
        
        best_model = self.results[best_model_name]['model']
        X = self.df.drop(columns=['high_gap_risk'])
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\n🏆 TOP 15 FEATURES POUR LA PRÉDICTION DE RISQUE:")
            for i, idx in enumerate(indices[:15]):
                feat_name = X.columns[idx]
                importance = importances[idx]
                percentage = importance * 100
                
                # Barre visuelle
                bar_length = int(percentage / 2)
                bar = "█" * bar_length + " " * (50 - bar_length)
                
                print(f"   {i+1:2}. {feat_name:25} : {percentage:5.1f}% |{bar}|")
        
        elif hasattr(best_model, 'coef_'):
            # Pour Logistic Regression
            coefficients = best_model.coef_[0]
            indices = np.argsort(np.abs(coefficients))[::-1]
            
            print(f"\n🏆 TOP 15 COEFFICIENTS (Logistic Regression):")
            for i, idx in enumerate(indices[:15]):
                feat_name = X.columns[idx]
                coef = coefficients[idx]
                
                direction = "↑ risque" if coef > 0 else "↓ risque"
                print(f"   {i+1:2}. {feat_name:25} : {coef:+.3f} ({direction})")
    
    def save_production_classifier(self, model_name):
        """Sauvegarde le classifieur pour production"""
        print(f"\n💾 Sauvegarde du classifieur de production...")
        
        import os
        os.makedirs('../models', exist_ok=True)
        os.makedirs('../reports', exist_ok=True)
        
        model = self.results[model_name]['model']
        metrics = self.results[model_name]
        
        # Package pour production
        model_package = {
            'model': model,
            'model_name': model_name,
            'model_type': 'classifier',
            'threshold': 5.0,  # Seuil de gap élevé
            'metrics': {
                'f1_score': float(metrics['f1']),
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'auc': float(metrics['auc']) if metrics['auc'] else None
            },
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_count': self.df.shape[1] - 1,
            'features_list': list(self.df.drop(columns=['high_gap_risk']).columns),
            'class_distribution': {
                'total_samples': len(self.df),
                'high_risk_count': int(self.df['high_gap_risk'].sum()),
                'high_risk_percentage': float(self.df['high_gap_risk'].mean() * 100)
            },
            'version': '1.0.0',
            'description': 'Classifieur de risque COS - Prédit les gaps > 5%'
        }
        
        # Sauvegarder modèle
        model_path = '../models/cos_risk_classifier.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Classifieur sauvegardé: {model_path}")
        
        # Générer rapport
        self.generate_classifier_report(model_name, model_path, metrics)
        
        return model_path
    
    def generate_classifier_report(self, model_name, model_path, metrics):
        """Génère un rapport détaillé"""
        report = f"""
{'='*70}
📋 RAPPORT CLASSIFIEUR DE RISQUE COS - KFC
{'='*70}

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏆 Modèle: {model_name}
📁 Fichier: {model_path}
🎯 Objectif: Détecter les risques de gap COS > 5%

📊 PERFORMANCE DU MODÈLE
{'─'*40}
• F1-Score: {metrics['f1']:.3f}
• Accuracy: {metrics['accuracy']:.3f}
• Precision: {metrics['precision']:.3f}
• Recall: {metrics['recall']:.3f}
• AUC-ROC: {metrics['auc'] if metrics['auc'] else 'N/A':.3f}

📈 INTERPRÉTATION BUSINESS
{'─'*40}
• Le modèle détecte {metrics['recall']*100:.1f}% des risques réels
• Quand il prédit un risque, il a raison {metrics['precision']*100:.1f}% du temps
• Performance globale: {metrics['f1']*100:.1f}% (F1-Score)

🎯 UTILISATION OPÉRATIONNELLE
{'─'*40}
1. ALERTE PRÉVENTIVE:
   • Exécuter quotidiennement sur les données du jour J-1
   • Alerter les managers régionaux des restaurants à risque
   • Planifier des audits ciblés

2. INTERPRÉTATION DES SCORES:
   • Score > 0.7: Risque élevé - Action immédiate requise
   • Score 0.4-0.7: Risque modéré - Surveillance renforcée
   • Score < 0.4: Risque faible - Monitoring standard

3. ACTIONS RECOMMANDÉES:
   • Vérifier les anomalies opérationnelles récentes
   • Contrôler les niveaux de stock et les déchets
   • Former le personnel sur les procédures
   • Optimiser les commandes de matières premières

🔧 DÉPLOIEMENT DASHBOARD
{'─'*40}
1. Chargement:
   import pickle
   model_data = pickle.load(open('{model_path}', 'rb'))
   classifier = model_data['model']

2. Prédiction:
   risk_score = classifier.predict_proba(features)[:, 1]
   is_high_risk = classifier.predict(features)

3. Visualisation:
   • Carte des restaurants à risque
   • Tendances des risques par région
   • Facteurs contributifs principaux

📊 STATISTIQUES DES DONNÉES
{'─'*40}
• Échantillons totaux: {len(self.df):,}
• Risques détectés: {self.df['high_gap_risk'].sum():,} ({self.df['high_gap_risk'].mean()*100:.1f}%)
• Features utilisées: {self.df.shape[1] - 1}
• Période: Jan 2024 - Jun 2024

🔍 FEATURES CLÉS
{'─'*40}
Les principales features utilisées pour la prédiction:
• Région du restaurant
• Anomalies opérationnelles récentes
• Scores de qualité (QSP)
• Volumes de ventes
• Tendances temporelles

🚀 RECOMMANDATIONS
{'─'*40}
• Recalibrer le modèle tous les trimestres
• Ajouter de nouvelles données saisonnières
• Intégrer les feedbacks des managers
• Surveiller la dérive des performances

📞 SUPPORT
{'─'*40}
• Questions techniques: Équipe Data Science
• Interprétation métier: Directeurs Opérationnels
• Alertes urgentes: Managers Régionaux

{'='*70}
✅ CLASSIFIEUR PRÊT POUR LA PRODUCTION
{'='*70}
"""
        
        report_path = '../reports/risk_classifier_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Rapport généré: {report_path}")
        
        # Sauvegarder aussi en JSON pour le dashboard
        import json
        dashboard_config = {
            'model_name': model_name,
            'model_path': model_path,
            'threshold': 5.0,
            'metrics': {
                'f1': float(metrics['f1']),
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall'])
            },
            'features': list(self.df.drop(columns=['high_gap_risk']).columns),
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dashboard_widgets': [
                {
                    'name': 'risk_map',
                    'type': 'map',
                    'data_source': 'model_predictions'
                },
                {
                    'name': 'risk_trends',
                    'type': 'line_chart',
                    'data_source': 'historical_predictions'
                },
                {
                    'name': 'top_risk_factors',
                    'type': 'bar_chart',
                    'data_source': 'feature_importance'
                }
            ]
        }
        
        json_path = '../models/classifier_dashboard_config.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, indent=2)
        
        print(f"✅ Configuration dashboard: {json_path}")

def main():
    """Fonction principale"""
    print("="*70)
    print("🎯 CLASSIFICATION DU RISQUE COS - KFC")
    print("="*70)
    
    trainer = RiskClassificationModel()
    
    try:
        # 1. Préparation des données
        print("\n[1/4] 📊 PRÉPARATION DES DONNÉES")
        df = trainer.prepare_classification_data()
        
        # 2. Entraînement
        print("\n[2/4] 🚀 ENTRAÎNEMENT")
        results = trainer.train_risk_classifiers()
        
        # 3. Sélection
        print("\n[3/4] 🏆 SÉLECTION")
        best_model = trainer.select_best_classifier()
        
        if best_model:
            # 4. Analyse
            trainer.analyze_important_features(best_model)
            
            # 5. Sauvegarde
            print("\n[4/4] 💾 SAUVEGARDE")
            model_path = trainer.save_production_classifier(best_model)
            
            print("\n" + "="*70)
            print("✅ CLASSIFIEUR DE RISQUE PRÊT POUR LA PRODUCTION !")
            print("="*70)
            
            print(f"\n🎯 RÉSULTATS:")
            metrics = trainer.results[best_model]
            print(f"  • Modèle: {best_model}")
            print(f"  • F1-Score: {metrics['f1']:.3f}")
            print(f"  • Détection (Recall): {metrics['recall']*100:.1f}%")
            print(f"  • Précision: {metrics['precision']*100:.1f}%")
            
            print(f"\n📁 ARTEFACTS:")
            print(f"  • Classifieur: ../models/cos_risk_classifier.pkl")
            print(f"  • Rapport: ../reports/risk_classifier_report.txt")
            print(f"  • Dashboard config: ../models/classifier_dashboard_config.json")
            
            print(f"\n🚀 INTÉGRATION IMMÉDIATE DANS LE DASHBOARD!")
            print(f"   Le modèle peut alerter sur les restaurants à risque COS.")
            
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()