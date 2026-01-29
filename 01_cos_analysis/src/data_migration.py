"""
Script de migration des données CSV vers la base de données SQLite
Version corrigée et optimisée
"""

import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import hashlib
import sys
import os

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent))

class COSDatabase:
    """Classe simplifiée pour la migration"""
    
    def __init__(self, db_path: str = "cos_kfc.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON")
    
    def create_tables(self):
        """Crée les tables si elles n'existent pas"""
        # Table restaurants
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS restaurants (
            restaurant_id INTEGER PRIMARY KEY,
            restaurant_name VARCHAR(100) NOT NULL,
            region VARCHAR(50),
            opening_date DATE,
            manager_name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Table product_families
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS product_families (
            product_family_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_family_name VARCHAR(50) NOT NULL UNIQUE,
            category VARCHAR(50),
            target_gap_percentage DECIMAL(5,2) DEFAULT 4.00
        )
        """)
        
        # Table transactions
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            restaurant_id INTEGER NOT NULL,
            product_family_id INTEGER NOT NULL,
            theoretical_unit_cost DECIMAL(10,2) NOT NULL,
            actual_unit_cost DECIMAL(10,2) NOT NULL,
            selling_price DECIMAL(10,2) NOT NULL,
            units_sold INTEGER NOT NULL,
            waste_kg DECIMAL(10,2) DEFAULT 0,
            qsp_score DECIMAL(3,1) CHECK (qsp_score >= 0 AND qsp_score <= 10),
            operational_anomaly BOOLEAN DEFAULT FALSE,
            revenue DECIMAL(15,2),
            theoretical_cos DECIMAL(10,4),
            actual_cos DECIMAL(10,4),
            cos_gap DECIMAL(10,4),
            gap_percentage DECIMAL(10,2),
            gap_severity VARCHAR(10),
            FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id),
            FOREIGN KEY (product_family_id) REFERENCES product_families(product_family_id)
        )
        """)
        
        # Table users
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) CHECK (role IN ('viewer', 'manager', 'admin')) DEFAULT 'viewer',
            full_name VARCHAR(100),
            email VARCHAR(100),
            restaurant_id INTEGER,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
        )
        """)
        
        # Créer les indexes
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_restaurant ON transactions(restaurant_id)")
        self.conn.commit()
        
        # Données initiales
        self._seed_initial_data()
    
    def _seed_initial_data(self):
        """Insère les données initiales"""
        # Familles de produits
        product_families = [
            ('Poulet', 'Principal'),
            ('Accompagnements', 'Side'),
            ('Boissons', 'Beverage'),
            ('Desserts', 'Dessert'),
            ('Menus', 'Combo')
        ]
        
        self.cursor.executemany(
            "INSERT OR IGNORE INTO product_families (product_family_name, category) VALUES (?, ?)",
            product_families
        )
        
        # Utilisateurs par défaut
        users = [
            ('admin', '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9', 'admin', 'Administrateur', 'admin@kfc.fr'),
            ('manager', 'd0e4b4ccbb5c8c6f9c7e8c9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0', 'manager', 'Manager Régional', 'manager@kfc.fr'),
            ('viewer', '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', 'viewer', 'Consultant', 'viewer@kfc.fr')
        ]
        
        self.cursor.executemany(
            "INSERT OR IGNORE INTO users (username, password_hash, role, full_name, email) VALUES (?, ?, ?, ?, ?)",
            users
        )
        
        self.conn.commit()
    
    def execute_query(self, query, params=()):
        """Exécute une requête SQL"""
        self.cursor.execute(query, params)
        self.conn.commit()
        return self.cursor.fetchall()
    
    def close(self):
        """Ferme la connexion"""
        self.conn.close()


class DataMigrator:
    """Classe pour migrer les données CSV vers SQLite"""
    
    def __init__(self, csv_dir: str = "../data/raw", db_path: str = "cos_kfc.db"):
        """
        Initialise le migrator
        
        Args:
            csv_dir: Dossier contenant les fichiers CSV
            db_path: Chemin vers la base de données SQLite
        """
        self.csv_dir = Path(csv_dir)
        self.db = COSDatabase(db_path)
        self.db.create_tables()
        
    def migrate_all(self):
        """Migre toutes les données"""
        print("🚀 Démarrage de la migration des données...")
        print("=" * 60)
        
        # 1. Migrer les restaurants
        restaurants_success = self.migrate_restaurants()
        
        if not restaurants_success:
            print("❌ Échec migration restaurants - Arrêt")
            return False
        
        # 2. Migrer les transactions
        transactions_success = self.migrate_transactions()
        
        if not transactions_success:
            print("⚠️  Migration transactions partielle")
        
        # 3. Afficher les statistiques
        self.show_statistics()
        
        print("=" * 60)
        print("✅ Migration terminée!")
        
        return restaurants_success and transactions_success
    
    def migrate_restaurants(self):
        """Migre les données restaurants - VERSION SIMPLIFIÉE"""
        print("📊 Migration des restaurants...")
        
        # Chercher le fichier restaurants
        restaurant_files = list(self.csv_dir.glob("*restaurant*.csv"))
        if not restaurant_files:
            print("❌ Aucun fichier restaurants trouvé")
            return False
        
        restaurant_file = restaurant_files[0]
        print(f"   📁 Fichier: {restaurant_file.name}")
        
        try:
            # Lire le CSV
            df = pd.read_csv(restaurant_file)
            print(f"   📋 {len(df)} restaurants trouvés")
            print(f"   🔍 Colonnes: {list(df.columns)}")
            
            # Nettoyer les colonnes - garder seulement ce dont on a besoin
            # Vérifier qu'on a au minimum restaurant_id et restaurant_name
            required_cols = ['restaurant_id', 'restaurant_name']
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ Colonne requise manquante: {col}")
                    return False
            
            # Préparer les données pour insertion
            restaurants_data = []
            regions = ['Nord', 'Sud', 'Est', 'Ouest', 'Centre', 'Nord-Est', 'Sud-Ouest', 'Île-de-France', 'Provence', 'Bretagne']
            managers = ['Jean Dupont', 'Marie Martin', 'Pierre Bernard', 'Sophie Petit', 'Luc Dubois',
                       'Thomas Blanc', 'Julie Roux', 'Michel Vert', 'Camille Noir', 'Paul Gris']
            
            for idx, row in df.iterrows():
                restaurant_id = int(row['restaurant_id'])
                restaurant_name = str(row['restaurant_name']).strip()
                
                # Ajouter des informations supplémentaires
                region = regions[idx % len(regions)] if 'region' not in df.columns else str(row.get('region', ''))
                manager = managers[idx % len(managers)] if 'manager_name' not in df.columns else str(row.get('manager_name', ''))
                
                # Date d'ouverture aléatoire entre 2010 et 2023
                opening_year = 2010 + (idx % 14)
                opening_date = f"{opening_year}-01-{15 + (idx % 15):02d}"
                
                restaurants_data.append((
                    restaurant_id,
                    restaurant_name,
                    region,
                    opening_date,
                    manager
                ))
            
            # Insérer dans la base de données
            print("   💾 Insertion dans la base de données...")
            
            # D'abord supprimer les anciennes données
            self.db.cursor.execute("DELETE FROM restaurants")
            
            # Insérer les nouvelles
            insert_query = """
            INSERT INTO restaurants 
            (restaurant_id, restaurant_name, region, opening_date, manager_name)
            VALUES (?, ?, ?, ?, ?)
            """
            
            self.db.cursor.executemany(insert_query, restaurants_data)
            self.db.conn.commit()
            
            print(f"✅ {len(restaurants_data)} restaurants migrés avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur migration restaurants: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def migrate_transactions(self):
        """Migre les données transactions - VERSION ROBUSTE"""
        print("\n📊 Migration des transactions...")
        
        # Chercher le fichier transactions
        transaction_files = list(self.csv_dir.glob("*transaction*.csv"))
        if not transaction_files:
            print("❌ Aucun fichier transactions trouvé")
            return False
        
        transaction_file = transaction_files[0]
        print(f"   📁 Fichier: {transaction_file.name}")
        
        try:
            # Lire le CSV en chunks
            chunk_size = 5000
            total_migrated = 0
            chunks_processed = 0
            
            print(f"   📖 Lecture par chunks de {chunk_size} lignes...")
            
            # Récupérer les restaurants et produits existants
            restaurants = self.db.execute_query("SELECT restaurant_id FROM restaurants")
            restaurant_ids = [str(r[0]) for r in restaurants]
            
            products = self.db.execute_query("SELECT product_family_id, product_family_name FROM product_families")
            product_map = {p[1]: p[0] for p in products}
            
            if not restaurant_ids:
                print("❌ Aucun restaurant dans la base - impossible de migrer les transactions")
                return False
            
            # Vider la table transactions avant migration
            self.db.cursor.execute("DELETE FROM transactions")
            self.db.conn.commit()
            
            # Lire le fichier CSV
            for chunk in pd.read_csv(transaction_file, chunksize=chunk_size):
                chunks_processed += 1
                print(f"   🔄 Traitement chunk {chunks_processed} ({len(chunk)} lignes)...")
                
                # Normaliser les noms de colonnes
                chunk = self._normalize_columns(chunk)
                
                # Vérifier les colonnes essentielles
                required_cols = ['date', 'restaurant_id', 'product_family', 
                                'theoretical_unit_cost', 'actual_unit_cost', 
                                'selling_price', 'units_sold']
                
                missing_cols = [col for col in required_cols if col not in chunk.columns]
                if missing_cols:
                    print(f"   ⚠️  Colonnes manquantes dans chunk: {missing_cols}")
                    continue
                
                # Filtrer les transactions avec des restaurant_id valides
                chunk['restaurant_id'] = chunk['restaurant_id'].astype(str)
                valid_mask = chunk['restaurant_id'].isin(restaurant_ids)
                
                if not valid_mask.any():
                    print("   ⚠️  Aucune transaction valide dans ce chunk")
                    continue
                
                chunk = chunk[valid_mask]
                chunk['restaurant_id'] = chunk['restaurant_id'].astype(int)
                
                # Gérer les familles de produits
                unique_products = chunk['product_family'].unique()
                for product in unique_products:
                    if product not in product_map:
                        # Ajouter la nouvelle famille
                        self.db.cursor.execute(
                            "INSERT OR IGNORE INTO product_families (product_family_name) VALUES (?)",
                            (product,)
                        )
                        self.db.conn.commit()
                        
                        # Récupérer le nouvel ID
                        result = self.db.cursor.execute(
                            "SELECT product_family_id FROM product_families WHERE product_family_name = ?",
                            (product,)
                        ).fetchone()
                        
                        if result:
                            product_map[product] = result[0]
                
                # Mapper les noms de produits aux IDs
                chunk['product_family_id'] = chunk['product_family'].map(product_map)
                
                # Supprimer les lignes avec produits non mappés
                chunk = chunk.dropna(subset=['product_family_id'])
                chunk['product_family_id'] = chunk['product_family_id'].astype(int)
                
                # Calculer les champs dérivés
                chunk = self._calculate_derived_fields(chunk)
                
                # Préparer les données pour insertion
                transactions_data = []
                for _, row in chunk.iterrows():
                    try:
                        transactions_data.append((
                            pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                            int(row['restaurant_id']),
                            int(row['product_family_id']),
                            float(row['theoretical_unit_cost']),
                            float(row['actual_unit_cost']),
                            float(row['selling_price']),
                            int(row['units_sold']),
                            float(row.get('waste_kg', 0)),
                            float(row.get('qsp_score', 8.0)),
                            bool(row.get('operational_anomaly', False)),
                            float(row.get('revenue', row['selling_price'] * row['units_sold'])),
                            float(row.get('theoretical_cos', 
                                (row['theoretical_unit_cost'] * row['units_sold']) / 
                                (row['selling_price'] * row['units_sold']))),
                            float(row.get('actual_cos',
                                (row['actual_unit_cost'] * row['units_sold']) / 
                                (row['selling_price'] * row['units_sold']))),
                            float(row.get('cos_gap', 0)),
                            float(row.get('gap_percentage', 0)),
                            row.get('gap_severity', 'Faible')
                        ))
                    except Exception as e:
                        print(f"      ⚠️  Erreur préparation ligne: {e}")
                        continue
                
                # Insérer les données
                if transactions_data:
                    insert_query = """
                    INSERT INTO transactions 
                    (date, restaurant_id, product_family_id, theoretical_unit_cost, actual_unit_cost,
                     selling_price, units_sold, waste_kg, qsp_score, operational_anomaly,
                     revenue, theoretical_cos, actual_cos, cos_gap, gap_percentage, gap_severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    try:
                        self.db.cursor.executemany(insert_query, transactions_data)
                        self.db.conn.commit()
                        total_migrated += len(transactions_data)
                        print(f"   ✅ {total_migrated} transactions migrées au total")
                    except Exception as e:
                        print(f"   ❌ Erreur insertion chunk: {e}")
                        self.db.conn.rollback()
            
            print(f"\n📊 Résumé migration transactions:")
            print(f"   • Chunks traités: {chunks_processed}")
            print(f"   • Transactions migrées: {total_migrated}")
            
            return total_migrated > 0
            
        except Exception as e:
            print(f"❌ Erreur migration transactions: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _normalize_columns(self, df):
        """Normalise les noms de colonnes"""
        column_mapping = {
            # Dates
            'Date': 'date', 'DATE': 'date',
            # Restaurants
            'Restaurant_ID': 'restaurant_id', 'RESTAURANT_ID': 'restaurant_id',
            'Restaurant_Name': 'restaurant_name', 'RESTAURANT_NAME': 'restaurant_name',
            # Produits
            'Product_Family': 'product_family', 'PRODUCT_FAMILY': 'product_family',
            'product': 'product_family', 'Product': 'product_family',
            # Coûts
            'Theoretical_Unit_Cost': 'theoretical_unit_cost', 'theoretical_cost': 'theoretical_unit_cost',
            'Actual_Unit_Cost': 'actual_unit_cost', 'actual_cost': 'actual_unit_cost',
            # Prix et quantités
            'Selling_Price': 'selling_price', 'price': 'selling_price', 'Price': 'selling_price',
            'Units_Sold': 'units_sold', 'quantity': 'units_sold', 'Quantity': 'units_sold',
            # Métriques
            'Revenue': 'revenue', 'REVENUE': 'revenue',
            'Waste_KG': 'waste_kg', 'waste': 'waste_kg', 'WASTE_KG': 'waste_kg',
            'QSP_Score': 'qsp_score', 'qsp': 'qsp_score', 'QSP': 'qsp_score',
            'Operational_Anomaly': 'operational_anomaly', 'anomaly': 'operational_anomaly',
            # Écarts
            'Gap_Percentage': 'gap_percentage', 'gap': 'gap_percentage', 'GAP_PERCENTAGE': 'gap_percentage',
            'Gap_Severity': 'gap_severity', 'severity': 'gap_severity'
        }
        
        # Renommer les colonnes
        df = df.rename(columns=lambda x: column_mapping.get(x, x))
        
        # S'assurer que les colonnes essentielles existent
        if 'date' not in df.columns and 'Date' in df.columns:
            df['date'] = df['Date']
        
        return df
    
    def _calculate_derived_fields(self, df):
        """Calcule les champs dérivés"""
        # Calculer le revenue si manquant
        if 'revenue' not in df.columns or df['revenue'].isnull().any():
            df['revenue'] = df['selling_price'] * df['units_sold']
        
        # Calculer COS théorique
        if 'theoretical_cos' not in df.columns or df['theoretical_cos'].isnull().any():
            df['theoretical_cos'] = (df['theoretical_unit_cost'] * df['units_sold']) / df['revenue']
        
        # Calculer COS réel
        if 'actual_cos' not in df.columns or df['actual_cos'].isnull().any():
            df['actual_cos'] = (df['actual_unit_cost'] * df['units_sold']) / df['revenue']
        
        # Calculer écart COS
        if 'cos_gap' not in df.columns or df['cos_gap'].isnull().any():
            df['cos_gap'] = df['actual_cos'] - df['theoretical_cos']
        
        # Calculer pourcentage d'écart
        if 'gap_percentage' not in df.columns or df['gap_percentage'].isnull().any():
            df['gap_percentage'] = (df['cos_gap'] / df['theoretical_cos']) * 100
        
        # Calculer sévérité
        if 'gap_severity' not in df.columns or df['gap_severity'].isnull().any():
            df['gap_severity'] = df['gap_percentage'].apply(
                lambda x: 'Faible' if x <= 4 else 'Moyen' if x <= 6 else 'Élevé'
            )
        
        # Valeurs par défaut
        if 'waste_kg' not in df.columns:
            df['waste_kg'] = 0
        
        if 'qsp_score' not in df.columns:
            df['qsp_score'] = 8.0
        
        if 'operational_anomaly' not in df.columns:
            df['operational_anomaly'] = False
        
        return df
    
    def show_statistics(self):
        """Affiche les statistiques de la base de données"""
        print("\n📊 STATISTIQUES DE LA BASE DE DONNÉES")
        print("=" * 50)
        
        # Compter les enregistrements par table
        tables = ['restaurants', 'product_families', 'transactions', 'users']
        
        for table in tables:
            try:
                result = self.db.execute_query(f"SELECT COUNT(*) FROM {table}")
                count = result[0][0] if result else 0
                print(f"   {table:20} : {count:>8,} enregistrements")
            except:
                print(f"   {table:20} : (erreur)")
        
        # Taille du fichier
        if os.path.exists('cos_kfc.db'):
            size_mb = os.path.getsize('cos_kfc.db') / (1024 * 1024)
            print(f"\n   📏 Taille de la base : {size_mb:.2f} MB")
        
        # Test d'une requête complexe
        print("\n   🔍 Test de requête (top 3 restaurants par écart):")
        try:
            query = """
            SELECT 
                r.restaurant_name,
                COUNT(t.transaction_id) as nb_transactions,
                ROUND(AVG(t.gap_percentage), 2) as avg_gap,
                ROUND(SUM(t.revenue), 0) as total_revenue
            FROM restaurants r
            LEFT JOIN transactions t ON r.restaurant_id = t.restaurant_id
            GROUP BY r.restaurant_id, r.restaurant_name
            ORDER BY avg_gap DESC
            LIMIT 3
            """
            
            result = self.db.execute_query(query)
            if result:
                for row in result:
                    print(f"      • {row[0]}: {row[2]}% d'écart, {row[1]:,} transactions, {row[3]:,.0f}€ CA")
            else:
                print("      (pas de données)")
        except Exception as e:
            print(f"      (erreur requête: {e})")
        
        print("=" * 50)


def main():
    """Fonction principale de migration"""
    print("=" * 60)
    print("🗄️  MIGRATION CSV → SQLite - PROJET COS KFC")
    print("=" * 60)
    
    # Créer un backup du fichier de base de données existant
    if os.path.exists('cos_kfc.db'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"backup_cos_kfc_{timestamp}.db"
        import shutil
        shutil.copy2('cos_kfc.db', backup_path)
        print(f"📦 Backup créé: {backup_path}")
    
    # Créer et exécuter le migrator
    migrator = DataMigrator()
    success = migrator.migrate_all()
    
    # Fermer la connexion
    migrator.db.close()
    
    if success:
        print("\n🎉 MIGRATION RÉUSSIE!")
        print("La base de données SQLite est maintenant opérationnelle.")
        print("\nProchaines étapes:")
        print("1. Tester avec: python test_db.py")
        print("2. Mettre à jour le dashboard pour utiliser la base de données")
        print("3. Ajouter des tests de performance")
    else:
        print("\n⚠️  MIGRATION PARTIELLEMENT ÉCHOUÉE")
        print("Vérifiez les messages d'erreur ci-dessus.")
    
    return success


if __name__ == "__main__":
    main()