"""
Générateur de données synthétiques SIMPLE et CONTRÔLÉ
Écart moyen cible: 4.0% exactement
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

class COSDataGeneratorSimple:
    """Version SIMPLE avec contrôle précis des écarts"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # CONTRÔLE PRÉCIS
        self.TARGET_GAP = 0.04  # 4.0% exactement
        
        # Restaurants simples
        self.restaurants = {
            1: "KFC Paris Louvre",
            2: "KFC Lyon Part-Dieu", 
            3: "KFC Marseille Vieux Port",
            4: "KFC Toulouse Capitole",
            5: "KFC Lille Centre",
            6: "KFC Bordeaux Quinconces",
            7: "KFC Nantes Erdre",
            8: "KFC Strasbourg Grande Île",
            9: "KFC Rennes Centre",
            10: "KFC Nice Promenade"
        }
        
        # Profils avec écart cible PRÉCIS
        self.restaurant_gap_targets = {
            1: 0.03,  # 3%
            2: 0.02,  # 2%
            3: 0.08,  # 8% (problématique)
            4: 0.025, # 2.5%
            5: 0.045, # 4.5%
            6: 0.035, # 3.5%
            7: 0.05,  # 5%
            8: 0.022, # 2.2%
            9: 0.055, # 5.5%
            10: 0.038  # 3.8%
        }
        
        # Produits avec légères variations
        self.product_families = {
            "Poulet Frit": {"base_cost": 4.20, "gap_variation": 0.01},
            "Sandwichs": {"base_cost": 2.80, "gap_variation": 0.005},
            "Burger Poulet": {"base_cost": 3.10, "gap_variation": 0.007},
            "Wraps & Salades": {"base_cost": 3.50, "gap_variation": 0.009},
            "Accompagnements": {"base_cost": 1.50, "gap_variation": 0.012},
            "Boissons": {"base_cost": 0.80, "gap_variation": 0.003},
            "Desserts": {"base_cost": 2.20, "gap_variation": 0.006},
            "Petits Déjeuners": {"base_cost": 2.50, "gap_variation": 0.004}
        }

    def generate_transactions(self, start_date="2024-01-01", n_days=180):
        """Génération SIMPLE et CONTRÔLÉE"""
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        
        data = []
        transaction_id = 10000
        
        print("🎯 GÉNÉRATION SIMPLE - CIBLE: 4.0%")
        print("=" * 50)
        
        for date in dates:
            for rest_id, rest_name in self.restaurants.items():
                # Écart cible pour CE restaurant
                restaurant_target = self.restaurant_gap_targets[rest_id]
                
                for product, info in self.product_families.items():
                    # Coût théorique fixe
                    theoretical_cost = info["base_cost"]
                    
                    # Écart FINAL = cible restaurant + variation produit ± petite variation aléatoire
                    product_variation = info["gap_variation"]
                    random_variation = np.random.uniform(-0.005, 0.005)
                    
                    gap_percentage = restaurant_target + product_variation + random_variation
                    
                    # Coût réel = théorique × (1 + écart)
                    actual_cost = theoretical_cost * (1 + gap_percentage)
                    
                    # Ventes réalistes
                    units_sold = np.random.randint(20, 80)
                    
                    # Gaspillage proportionnel à l'écart
                    waste_kg = max(0, gap_percentage * 0.5 * units_sold * 0.05)
                    
                    # Prix de vente (marge standard)
                    selling_price = theoretical_cost * 2.2
                    
                    # Anomalie si écart > 7%
                    has_anomaly = gap_percentage > 0.07
                    
                    transaction_id += 1
                    
                    data.append({
                        "transaction_id": transaction_id,
                        "date": date.date(),
                        "restaurant_id": rest_id,
                        "restaurant_name": rest_name,
                        "product_family": product,
                        "theoretical_unit_cost": round(theoretical_cost, 2),
                        "actual_unit_cost": round(actual_cost, 2),
                        "selling_price": round(selling_price, 2),
                        "units_sold": units_sold,
                        "revenue": round(selling_price * units_sold, 2),
                        "waste_kg": round(waste_kg, 3),
                        "qsp_score": round(np.random.uniform(0.75, 0.95), 2),
                        "operational_anomaly": has_anomaly,
                        "calculated_gap_percentage": round(gap_percentage * 100, 2)
                    })
        
        df = pd.DataFrame(data)
        
        # Vérification et ajustement FINAL
        current_mean = df["calculated_gap_percentage"].mean()
        adjustment = self.TARGET_GAP * 100 - current_mean
        
        print(f"📊 Avant ajustement: {current_mean:.2f}%")
        print(f"📊 Ajustement nécessaire: {adjustment:.2f}%")
        
        # Ajustement LINÉAIRE pour atteindre exactement 4.0%
        if abs(adjustment) > 0.1:
            df["calculated_gap_percentage"] += adjustment
            df["actual_unit_cost"] = df["theoretical_unit_cost"] * (1 + df["calculated_gap_percentage"]/100)
            print(f"📊 Après ajustement: {df['calculated_gap_percentage'].mean():.2f}%")
        
        # Calcul COS final
        df["theoretical_cos"] = (df["theoretical_unit_cost"] * df["units_sold"]) / df["revenue"]
        df["actual_cos"] = (df["actual_unit_cost"] * df["units_sold"]) / df["revenue"]
        df["cos_gap"] = df["actual_cos"] - df["theoretical_cos"]
        df["gap_percentage"] = (df["cos_gap"] / df["theoretical_cos"]) * 100
        
        # Sévérité
        conditions = [
            df["gap_percentage"] > 10,
            df["gap_percentage"] > 5,
            df["gap_percentage"] > 2
        ]
        choices = ["CRITIQUE", "ÉLEVÉ", "MODÉRÉ"]
        df["gap_severity"] = np.select(conditions, choices, default="FAIBLE")
        
        return df

    def generate_restaurant_info(self):
        """Infos restaurants simples"""
        data = []
        for rest_id, rest_name in self.restaurants.items():
            target_gap = self.restaurant_gap_targets[rest_id]
            
            # Classification performance
            if target_gap < 0.03:
                perf = "Excellent"
                audit = np.random.uniform(0.90, 0.96)
            elif target_gap < 0.045:
                perf = "Bon"
                audit = np.random.uniform(0.85, 0.92)
            elif target_gap < 0.06:
                perf = "Moyen"
                audit = np.random.uniform(0.78, 0.88)
            else:
                perf = "À améliorer"
                audit = np.random.uniform(0.70, 0.82)
            
            data.append({
                "restaurant_id": rest_id,
                "restaurant_name": rest_name,
                "performance_category": perf,
                "target_gap_percentage": round(target_gap * 100, 2),
                "manager_experience": np.random.randint(12, 60),
                "team_size": np.random.randint(20, 30),
                "last_audit_score": round(audit, 2),
                "monthly_target": np.random.uniform(100000, 150000)
            })
        
        return pd.DataFrame(data)

def main():
    """Exécution finale"""
    print("=" * 60)
    print("🎯 GÉNÉRATEUR FINAL - CONTRÔLE PRÉCIS 4.0%")
    print("=" * 60)
    
    generator = COSDataGeneratorSimple()
    
    # Génération
    transactions = generator.generate_transactions()
    restaurants = generator.generate_restaurant_info()
    
    # Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("../data/raw", exist_ok=True)
    
    trans_path = f"../data/raw/transactions_final_{timestamp}.csv"
    rest_path = f"../data/raw/restaurants_final_{timestamp}.csv"
    
    transactions.to_csv(trans_path, index=False)
    restaurants.to_csv(rest_path, index=False)
    
    # Statistiques FINALES
    print(f"\n✅ DONNÉES FINALES GÉNÉRÉES !")
    print(f"📁 Transactions: {trans_path}")
    print(f"📁 Restaurants: {rest_path}")
    
    print(f"\n📊 RÉSULTAT FINAL:")
    print(f"• Écart moyen: {transactions['gap_percentage'].mean():.2f}%")
    print(f"• Médiane: {transactions['gap_percentage'].median():.2f}%")
    print(f"• Std: {transactions['gap_percentage'].std():.2f}%")
    print(f"• Min/Max: {transactions['gap_percentage'].min():.2f}% / {transactions['gap_percentage'].max():.2f}%")
    
    print(f"\n📈 DISTRIBUTION:")
    dist = transactions['gap_severity'].value_counts(normalize=True).sort_index()
    for severity, pct in dist.items():
        print(f"• {severity:10}: {pct*100:5.1f}%")
    
    print(f"\n🏪 PERFORMANCE RESTAURANTS:")
    restaurant_stats = transactions.groupby('restaurant_name').agg({
        'gap_percentage': 'mean',
        'operational_anomaly': 'mean'
    }).round(3).sort_values('gap_percentage')
    
    print(restaurant_stats.head(3))
    print("...")
    print(restaurant_stats.tail(3))
    
    print(f"\n🎯 OBJECTIF ATTEINT: {'✅ OUI' if 3.8 <= transactions['gap_percentage'].mean() <= 4.2 else '❌ NON'}")
    
    # Créer aussi une version simplifiée pour exploration
    simple_trans = transactions[[
        'transaction_id', 'date', 'restaurant_name', 'product_family',
        'gap_percentage', 'gap_severity', 'operational_anomaly'
    ]].copy()
    
    simple_path = f"../data/processed/simple_for_analysis_{timestamp}.csv"
    simple_trans.to_csv(simple_path, index=False)
    print(f"\n💾 Version simplifiée: {simple_path}")

if __name__ == "__main__":
    main()