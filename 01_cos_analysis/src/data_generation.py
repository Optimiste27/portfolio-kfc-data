"""
Générateur de données synthétiques réalistes pour l'analyse COS
Simule les écarts de coût dans la restauration rapide
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

class COSDataGenerator:
    """Génère des données synthétiques réalistes pour l'analyse COS"""
    
    def __init__(self, seed=42):
        """Initialise le générateur avec une seed pour reproductibilité"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Configuration des restaurants
        self.restaurants = {
            1: {"name": "KFC Paris Louvre", "city": "Paris", "open_date": "2020-03-15"},
            2: {"name": "KFC Lyon Part-Dieu", "city": "Lyon", "open_date": "2019-08-22"},
            3: {"name": "KFC Marseille Vieux Port", "city": "Marseille", "open_date": "2021-01-10"},
            4: {"name": "KFC Toulouse Capitole", "city": "Toulouse", "open_date": "2018-11-30"},
            5: {"name": "KFC Lille Centre", "city": "Lille", "open_date": "2022-05-18"},
            6: {"name": "KFC Bordeaux Quinconces", "city": "Bordeaux", "open_date": "2019-04-12"},
            7: {"name": "KFC Nantes Erdre", "city": "Nantes", "open_date": "2020-09-05"},
            8: {"name": "KFC Strasbourg Grande Île", "city": "Strasbourg", "open_date": "2021-07-20"},
            9: {"name": "KFC Rennes Centre", "city": "Rennes", "open_date": "2022-02-28"},
            10: {"name": "KFC Nice Promenade", "city": "Nice", "open_date": "2018-06-14"}
        }
        
        # Familles de produits avec coûts de base réalistes
        self.product_families = {
            "Poulet Frit": {
                "base_cost": 4.20,
                "margin_multiplier": 2.1,
                "waste_factor": 0.15,
                "cost_variability": 0.12,
                "sensitivity": "high"
            },
            "Sandwichs": {
                "base_cost": 2.80,
                "margin_multiplier": 2.3,
                "waste_factor": 0.08,
                "cost_variability": 0.08,
                "sensitivity": "medium"
            },
            "Burger Poulet": {
                "base_cost": 3.10,
                "margin_multiplier": 2.2,
                "waste_factor": 0.10,
                "cost_variability": 0.10,
                "sensitivity": "medium"
            },
            "Wraps & Salades": {
                "base_cost": 3.50,
                "margin_multiplier": 2.0,
                "waste_factor": 0.12,
                "cost_variability": 0.09,
                "sensitivity": "medium"
            },
            "Accompagnements": {
                "base_cost": 1.50,
                "margin_multiplier": 2.5,
                "waste_factor": 0.20,
                "cost_variability": 0.05,
                "sensitivity": "low"
            },
            "Boissons": {
                "base_cost": 0.80,
                "margin_multiplier": 3.0,
                "waste_factor": 0.03,
                "cost_variability": 0.02,
                "sensitivity": "very_low"
            },
            "Desserts": {
                "base_cost": 2.20,
                "margin_multiplier": 2.4,
                "waste_factor": 0.09,
                "cost_variability": 0.07,
                "sensitivity": "medium"
            },
            "Petits Déjeuners": {
                "base_cost": 2.50,
                "margin_multiplier": 2.2,
                "waste_factor": 0.07,
                "cost_variability": 0.06,
                "sensitivity": "low"
            }
        }
        
        # Variantes par famille
        self.product_variants = {
            "Poulet Frit": ["Bucket 6 pièces", "Bucket 10 pièces", "Tenders 8p", "Ailes 12p"],
            "Sandwichs": ["Original", "Spicy", "Cheese", "Bacon"],
            "Burger Poulet": ["Classic", "Bacon Cheese", "Deluxe", "Spicy"],
            "Wraps & Salades": ["Caesar Wrap", "Spicy Wrap", "Salade Caesar", "Salade Poulet"],
            "Accompagnements": ["Frites Moyenne", "Frites Grande", "Potatoes", "Coleslaw"],
            "Boissons": ["Coca 33cl", "Coca 50cl", "Fanta", "Eau 50cl"],
            "Desserts": ["Cookie", "Brownie", "Sundae Chocolat", "Sundae Caramel"],
            "Petits Déjeuners": ["Croissant", "Pain au Chocolat", "Petit Déj Complet"]
        }
        
        # Types d'anomalies opérationnelles
        self.anomaly_types = [
            "surconsommation",
            "sous_cuisson",
            "surcuisson",
            "vol",
            "erreur_commande",
            "panne_equipement",
            "rupture_stock",
            "formation_manquante"
        ]

    def _calculate_seasonal_factor(self, date):
        """Calcule un facteur saisonnier basé sur la date"""
        day_of_week = date.weekday()
        month = date.month
        
        # Weekend = plus de ventes
        if day_of_week >= 5:
            sales_factor = np.random.uniform(1.15, 1.35)
        else:
            sales_factor = np.random.uniform(0.9, 1.1)
        
        # Vacances scolaires (simplifié)
        if month in [7, 8]:  # Juillet-Août
            sales_factor *= np.random.uniform(1.1, 1.25)
        elif month in [12]:  # Décembre
            sales_factor *= np.random.uniform(1.05, 1.15)
        
        return sales_factor

    def _get_restaurant_performance_factor(self, restaurant_id):
        """Facteur de performance basé sur l'ID restaurant (simulé)"""
        if restaurant_id % 2 == 0:
            return np.random.uniform(0.95, 1.0)
        else:
            return np.random.uniform(1.0, 1.05)

    def _simulate_operational_anomaly(self):
        """Simule une anomalie opérationnelle réaliste"""
        if np.random.random() < 0.1:
            anomaly_type = random.choice(self.anomaly_types)
            
            impacts = {
                "surconsommation": {"cost_multiplier": 1.25, "waste_multiplier": 1.0},
                "sous_cuisson": {"cost_multiplier": 1.0, "waste_multiplier": 1.3},
                "surcuisson": {"cost_multiplier": 1.0, "waste_multiplier": 1.4},
                "vol": {"cost_multiplier": 1.15, "waste_multiplier": 1.0},
                "erreur_commande": {"cost_multiplier": 1.1, "waste_multiplier": 1.2},
                "panne_equipement": {"cost_multiplier": 1.05, "waste_multiplier": 1.5},
                "rupture_stock": {"cost_multiplier": 1.0, "waste_multiplier": 1.0},
                "formation_manquante": {"cost_multiplier": 1.2, "waste_multiplier": 1.1}
            }
            
            return True, anomaly_type, impacts[anomaly_type]
        
        return False, None, {"cost_multiplier": 1.0, "waste_multiplier": 1.0}

    def _calculate_qsp_score(self, restaurant_id, has_anomaly):
        """Calcule un score QSP réaliste"""
        base_score = np.random.uniform(0.85, 0.95)
        
        if restaurant_id in [2, 4, 8]:
            base_score += 0.03
        elif restaurant_id in [3, 7]:
            base_score -= 0.02
        
        if has_anomaly:
            base_score -= np.random.uniform(0.05, 0.15)
        
        return max(0.5, min(1.0, base_score))

    def generate_daily_transactions(self, start_date="2024-01-01", n_days=180):
        """Génère les transactions quotidiennes"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        dates = [start + timedelta(days=i) for i in range(n_days)]
        
        data = []
        transaction_id = 10000
        
        print(f"Génération de données pour {n_days} jours...")
        print(f"Restaurants: {len(self.restaurants)}")
        print(f"Produits: {len(self.product_families)} familles")
        
        for date in dates:
            for restaurant_id in self.restaurants.keys():
                seasonal_factor = self._calculate_seasonal_factor(date)
                perf_factor = self._get_restaurant_performance_factor(restaurant_id)
                
                for product_family, product_info in self.product_families.items():
                    has_anomaly, anomaly_type, anomaly_impact = self._simulate_operational_anomaly()
                    
                    base_cost = product_info["base_cost"]
                    cost_variability = product_info["cost_variability"]
                    
                    theoretical_cost = np.random.normal(base_cost, base_cost * cost_variability * 0.5)
                    theoretical_cost = max(base_cost * 0.8, min(theoretical_cost, base_cost * 1.2))
                    
                    actual_cost = theoretical_cost * anomaly_impact["cost_multiplier"]
                    actual_cost *= perf_factor
                    
                    base_units = np.random.randint(15, 60)
                    units_sold = int(base_units * seasonal_factor * perf_factor)
                    
                    base_waste = product_info["waste_factor"] * units_sold * 0.05
                    waste_kg = base_waste * anomaly_impact["waste_multiplier"]
                    
                    selling_price = theoretical_cost * product_info["margin_multiplier"]
                    
                    qsp_score = self._calculate_qsp_score(restaurant_id, has_anomaly)
                    
                    variant = random.choice(self.product_variants[product_family])
                    
                    shift_type = random.choices(["midi", "soir"], weights=[0.6, 0.4])[0]
                    
                    transaction_id += 1
                    
                    data.append({
                        "transaction_id": transaction_id,
                        "date": date.date(),
                        "restaurant_id": restaurant_id,
                        "restaurant_name": self.restaurants[restaurant_id]["name"],
                        "product_family": product_family,
                        "product_variant": variant,
                        "theoretical_unit_cost": round(theoretical_cost, 2),
                        "actual_unit_cost": round(actual_cost, 2),
                        "selling_price": round(selling_price, 2),
                        "units_sold": units_sold,
                        "waste_kg": round(waste_kg, 2),
                        "qsp_score": round(qsp_score, 2),
                        "operational_anomaly": has_anomaly,
                        "anomaly_type": anomaly_type if has_anomaly else None,
                        "shift_type": shift_type,
                        "weather_impact": round(np.random.uniform(0.9, 1.15), 2)
                    })
        
        df = pd.DataFrame(data)
        
        df["revenue"] = df["selling_price"] * df["units_sold"]
        df["theoretical_cos"] = (df["theoretical_unit_cost"] * df["units_sold"]) / df["revenue"]
        df["actual_cos"] = (df["actual_unit_cost"] * df["units_sold"]) / df["revenue"]
        df["cos_gap"] = df["actual_cos"] - df["theoretical_cos"]
        df["gap_percentage"] = (df["cos_gap"] / df["theoretical_cos"]) * 100
        df["waste_cost"] = df["waste_kg"] * df["theoretical_unit_cost"]
        
        conditions = [
            df["gap_percentage"] > 10,
            df["gap_percentage"] > 5,
            df["gap_percentage"] > 2
        ]
        choices = ["CRITIQUE", "ÉLEVÉ", "MODÉRÉ"]
        df["gap_severity"] = np.select(conditions, choices, default="FAIBLE")
        
        return df

    def generate_restaurant_info(self):
        """Génère les informations des restaurants"""
        data = []
        
        for restaurant_id, info in self.restaurants.items():
            if restaurant_id % 2 == 0:
                experience = np.random.randint(24, 48)
            else:
                experience = np.random.randint(6, 24)
            
            open_date = datetime.strptime(info["open_date"], "%Y-%m-%d")
            equipment_age = (datetime.now() - open_date).days // 30
            
            data.append({
                "restaurant_id": restaurant_id,
                "restaurant_name": info["name"],
                "city": info["city"],
                "opening_date": info["open_date"],
                "monthly_sales_target": np.random.uniform(80000, 150000),
                "manager_experience": experience,
                "team_size": np.random.randint(15, 30),
                "equipment_age": equipment_age,
                "last_audit_score": round(np.random.uniform(0.8, 0.98), 2)
            })
        
        return pd.DataFrame(data)

    def save_data(self, output_dir="../data"):
        """Génère et sauvegarde toutes les données"""
        raw_dir = os.path.join(output_dir, "raw")
        processed_dir = os.path.join(output_dir, "processed")
        
        for directory in [raw_dir, processed_dir]:
            os.makedirs(directory, exist_ok=True)
        
        print("Génération des transactions...")
        transactions_df = self.generate_daily_transactions()
        
        print("Génération des infos restaurants...")
        restaurant_df = self.generate_restaurant_info()
        
        transactions_path = os.path.join(raw_dir, "transactions_daily.csv")
        restaurant_path = os.path.join(raw_dir, "restaurant_info.csv")
        
        transactions_df.to_csv(transactions_path, index=False)
        restaurant_df.to_csv(restaurant_path, index=False)
        
        print(f"\n✅ Données générées avec succès !")
        print(f"📁 Transactions: {transactions_path}")
        print(f"📁 Restaurants: {restaurant_path}")
        print(f"📊 Statistiques:")
        print(f"   - {len(transactions_df):,} transactions")
        print(f"   - Période: {transactions_df['date'].min()} to {transactions_df['date'].max()}")
        print(f"   - Restaurants: {transactions_df['restaurant_id'].nunique()}")
        print(f"   - Écart COS moyen: {transactions_df['gap_percentage'].mean():.2f}%")
        print(f"   - Anomalies détectées: {transactions_df['operational_anomaly'].sum():,}")
        
        return transactions_df, restaurant_df

def main():
    """Point d'entrée principal"""
    print("=" * 60)
    print("GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES - ANALYSE COS")
    print("=" * 60)
    
    generator = COSDataGenerator(seed=42)
    
    transactions_df, restaurant_df = generator.save_data()
    
    print("\n📋 APERÇU DES DONNÉES:")
    print(transactions_df.head())
    
    print("\n📊 DISTRIBUTION DES ÉCARTS:")
    print(transactions_df["gap_severity"].value_counts())
    
    print("\n🏪 PERFORMANCE PAR RESTAURANT (TOP 3 écarts):")
    restaurant_stats = transactions_df.groupby("restaurant_name").agg({
        "gap_percentage": "mean",
        "operational_anomaly": "sum"
    }).round(2)
    print(restaurant_stats.sort_values("gap_percentage", ascending=False).head(3))
    
    print("\n🎯 Génération terminée !")

if __name__ == "__main__":
    main()