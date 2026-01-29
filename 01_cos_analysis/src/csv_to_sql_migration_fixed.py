# src/csv_to_sql_migration_fixed.py
import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np

def create_connection(db_file):
    """Crée une connexion à la base SQLite"""
    conn = sqlite3.connect(db_file)
    return conn

def migrate_restaurants(csv_path, conn):
    """Migre les restaurants depuis CSV vers la base"""
    print("Migration des restaurants...")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Extraire la région depuis le nom
    def extract_region(name):
        parts = name.split()
        if len(parts) > 2:
            return parts[1]  # "KFC Paris Louvre" -> "Paris"
        return 'Inconnu'
    
    df['region'] = df['restaurant_name'].apply(extract_region)
    
    # Préparer les données
    restaurants_data = []
    for _, row in df.iterrows():
        restaurants_data.append((
            int(row['restaurant_id']),
            row['restaurant_name'],
            row['region'],
            '2020-01-01',  # opening_date par défaut
            f"Manager {row['restaurant_name']}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT OR REPLACE INTO restaurants 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', restaurants_data)
    
    conn.commit()
    print(f"✓ {len(restaurants_data)} restaurants migrés")

def create_product_families(conn, transactions_csv_path):
    """Crée les familles de produits avec nettoyage des noms"""
    print("Création des familles de produits...")
    
    df = pd.read_csv(transactions_csv_path, encoding='utf-8')
    
    # Nettoyer les noms de produits
    df['product_family_clean'] = df['product_family'].str.strip()
    
    # Obtenir les produits uniques
    product_families = df['product_family_clean'].unique()
    print(f"Produits uniques trouvés: {len(product_families)}")
    
    # Catégoriser les produits
    def categorize_product(product_name):
        product_lower = product_name.lower()
        if 'poulet' in product_lower and 'frit' in product_lower:
            return 'Poulet', 4.0
        elif 'poulet' in product_lower:
            return 'Poulet', 4.0
        elif 'sandwich' in product_lower or 'burger' in product_lower:
            return 'Sandwich', 3.5
        elif 'wrap' in product_lower or 'salade' in product_lower:
            return 'Sandwich', 3.5
        elif 'frite' in product_lower or 'potato' in product_lower:
            return 'Accompagnement', 2.5
        elif 'dessert' in product_lower:
            return 'Dessert', 5.0
        elif 'boisson' in product_lower:
            return 'Boisson', 1.5
        else:
            return 'Autre', 4.0
    
    cursor = conn.cursor()
    
    # Vider et recréer la table
    cursor.execute('DELETE FROM product_families')
    
    # Insérer chaque produit
    product_mapping = {}
    for i, product in enumerate(sorted(product_families), 1):
        category, target_gap = categorize_product(product)
        
        cursor.execute('''
            INSERT INTO product_families 
            (product_family_id, product_family_name, category, target_gap_percentage)
            VALUES (?, ?, ?, ?)
        ''', (i, product, category, target_gap))
        
        product_mapping[product] = i
    
    conn.commit()
    print(f"✓ {len(product_families)} familles de produits créées")
    return product_mapping

def migrate_transactions(csv_path, conn, product_family_mapping):
    """Migre les transactions avec nettoyage des produits"""
    print("Migration des transactions...")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Nettoyer les noms de produits
    df['product_family_clean'] = df['product_family'].str.strip()
    
    # Vérifier les produits manquants
    missing_products = set(df['product_family_clean']) - set(product_family_mapping.keys())
    if missing_products:
        print(f"Produits non mappés trouvés: {len(missing_products)}")
        for product in sorted(missing_products):
            print(f"  - {product}")
    
    # Préparer les données
    transactions_data = []
    missing_count = 0
    
    for _, row in df.iterrows():
        try:
            # Trouver l'ID du produit
            product_clean = row['product_family_clean']
            product_family_id = product_family_mapping.get(product_clean)
            
            if not product_family_id:
                missing_count += 1
                if missing_count <= 5:  # Afficher seulement les 5 premiers
                    print(f"  Produit non trouvé dans le mapping: {product_clean}")
                continue
            
            # Nettoyer les valeurs
            operational_anomaly = 1 if str(row.get('operational_anomaly', 'False')).lower() == 'true' else 0
            
            # Vérifier les valeurs numériques
            waste_kg = float(row['waste_kg']) if pd.notna(row.get('waste_kg')) else 0.0
            qsp_score = float(row['qsp_score']) if pd.notna(row.get('qsp_score')) else 0.0
            
            transactions_data.append((
                int(row['transaction_id']),
                row['date'],
                int(row['restaurant_id']),
                product_family_id,
                float(row['theoretical_unit_cost']),
                float(row['actual_unit_cost']),
                float(row['selling_price']),
                int(row['units_sold']),
                waste_kg,
                qsp_score,
                operational_anomaly,
                float(row['revenue']),
                float(row['theoretical_cos']),
                float(row['actual_cos']),
                float(row['cos_gap']),
                float(row['gap_percentage']),
                row['gap_severity']
            ))
        except Exception as e:
            print(f"Erreur ligne {_}: {e}")
            continue
    
    if missing_count > 5:
        print(f"  ... et {missing_count - 5} autres produits non mappés")
    
    # Insérer en batch
    cursor = conn.cursor()
    
    # Supprimer les anciennes transactions
    cursor.execute('DELETE FROM transactions')
    
    # Insérer les nouvelles
    cursor.executemany('''
        INSERT INTO transactions 
        (transaction_id, date, restaurant_id, product_family_id, 
         theoretical_unit_cost, actual_unit_cost, selling_price, units_sold,
         waste_kg, qsp_score, operational_anomaly, revenue,
         theoretical_cos, actual_cos, cos_gap, gap_percentage, gap_severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', transactions_data)
    
    conn.commit()
    print(f"✓ {len(transactions_data)} transactions migrées ({missing_count} ignorées)")

def create_kpi_daily(conn):
    """Recalcule les KPIs journaliers"""
    print("Création des KPIs journaliers...")
    
    cursor = conn.cursor()
    
    # Vider la table
    cursor.execute('DELETE FROM kpi_daily')
    
    # Recréer les KPIs
    cursor.execute('''
        INSERT INTO kpi_daily
        (kpi_date, restaurant_id, total_transactions, avg_gap_percentage, 
         total_revenue, total_waste_cost, anomaly_rate)
        SELECT 
            date,
            restaurant_id,
            COUNT(*) as total_transactions,
            ROUND(AVG(gap_percentage), 2) as avg_gap_percentage,
            ROUND(SUM(revenue), 2) as total_revenue,
            ROUND(SUM(waste_kg * theoretical_unit_cost), 2) as total_waste_cost,
            ROUND(AVG(CASE WHEN operational_anomaly = 1 THEN 1 ELSE 0 END) * 100, 2) as anomaly_rate
        FROM transactions
        GROUP BY date, restaurant_id
        ORDER BY date, restaurant_id
    ''')
    
    conn.commit()
    
    cursor.execute('SELECT COUNT(*) FROM kpi_daily')
    count = cursor.fetchone()[0]
    print(f"✓ {count} enregistrements KPI créés")

def create_sample_users(conn):
    """Crée des utilisateurs de test"""
    print("Création des utilisateurs de test...")
    
    cursor = conn.cursor()
    
    # Ajouter quelques utilisateurs
    users = [
        (1, 'admin', 'pbkdf2:sha256:260000$...', 'admin', 'Administrateur Système', 'admin@kfc.com', None, 1, None),
        (2, 'manager1', 'pbkdf2:sha256:260000$...', 'manager', 'Jean Dupont', 'jean.dupont@kfc.com', 1, 1, None),
        (3, 'viewer1', 'pbkdf2:sha256:260000$...', 'viewer', 'Marie Curie', 'marie.curie@kfc.com', None, 1, None)
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO users 
        (user_id, username, password_hash, role, full_name, email, restaurant_id, is_active, last_login)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', users)
    
    conn.commit()
    print("✓ 3 utilisateurs de test créés")

def create_views(conn):
    """Crée des vues pour faciliter les requêtes"""
    print("Création des vues SQL...")
    
    cursor = conn.cursor()
    
    # Vue pour les transactions détaillées
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS v_transactions_detailed AS
        SELECT 
            t.*,
            r.restaurant_name,
            r.region,
            p.product_family_name,
            p.category as product_category,
            p.target_gap_percentage as product_target_gap,
            (t.actual_unit_cost - t.theoretical_unit_cost) as cost_variance,
            (t.revenue - (t.actual_unit_cost * t.units_sold)) as profit,
            ROUND((t.revenue - (t.actual_unit_cost * t.units_sold)) / t.revenue * 100, 2) as profit_margin
        FROM transactions t
        LEFT JOIN restaurants r ON t.restaurant_id = r.restaurant_id
        LEFT JOIN product_families p ON t.product_family_id = p.product_family_id
    ''')
    
    # Vue pour les performances quotidiennes par restaurant
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS v_daily_performance AS
        SELECT 
            k.*,
            r.restaurant_name,
            r.region,
            ROUND(k.total_revenue / k.total_transactions, 2) as avg_transaction_value,
            CASE 
                WHEN k.avg_gap_percentage <= 2 THEN 'Excellent'
                WHEN k.avg_gap_percentage <= 4 THEN 'Bon'
                WHEN k.avg_gap_percentage <= 6 THEN 'Modéré'
                ELSE 'À améliorer'
            END as performance_category
        FROM kpi_daily k
        LEFT JOIN restaurants r ON k.restaurant_id = r.restaurant_id
    ''')
    
    conn.commit()
    print("✓ 2 vues SQL créées")

def validate_migration(conn):
    """Valide l'intégrité des données"""
    print("\n=== VALIDATION FINALE ===")
    
    cursor = conn.cursor()
    
    # Statistiques
    tables = ['restaurants', 'product_families', 'transactions', 'kpi_daily', 'users']
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {table}')
        count = cursor.fetchone()[0]
        print(f"{table}: {count} enregistrements")
    
    # Vérifications critiques
    print("\nVérifications d'intégrité:")
    
    # 1. Toutes les transactions ont un restaurant valide
    cursor.execute('''
        SELECT COUNT(*) 
        FROM transactions t
        LEFT JOIN restaurants r ON t.restaurant_id = r.restaurant_id
        WHERE r.restaurant_id IS NULL
    ''')
    orphan_transactions = cursor.fetchone()[0]
    print(f"✓ Transactions orphelines: {orphan_transactions}")
    
    # 2. Toutes les transactions ont un produit valide
    cursor.execute('''
        SELECT COUNT(*) 
        FROM transactions t
        LEFT JOIN product_families p ON t.product_family_id = p.product_family_id
        WHERE p.product_family_id IS NULL
    ''')
    invalid_products = cursor.fetchone()[0]
    print(f"✓ Transactions avec produits invalides: {invalid_products}")
    
    # 3. Vérifier la plage de dates
    cursor.execute('SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM transactions')
    min_date, max_date, unique_dates = cursor.fetchone()
    print(f"✓ Période: {min_date} à {max_date} ({unique_dates} jours)")
    
    # 4. Vérifier les restaurants actifs
    cursor.execute('SELECT COUNT(DISTINCT restaurant_id) FROM transactions')
    active_restaurants = cursor.fetchone()[0]
    print(f"✓ Restaurants avec transactions: {active_restaurants}")
    
    # 5. Vérifier les revenus totaux
    cursor.execute('SELECT ROUND(SUM(revenue), 2) FROM transactions')
    total_revenue = cursor.fetchone()[0]
    print(f"✓ Revenu total: {total_revenue:,.2f} €")
    
    return orphan_transactions == 0 and invalid_products == 0

def main():
    """Migration complète avec correction"""
    print("=== MIGRATION CSV → SQLite (VERSION CORRIGÉE) ===")
    
    # Chemins
    db_path = 'cos_kfc.db'
    restaurants_csv = '../data/raw/restaurants_final_20260124_1112.csv'
    transactions_csv = '../data/raw/transactions_final_20260124_1112.csv'
    
    # Connexion
    conn = create_connection(db_path)
    
    try:
        # 1. Migrer les restaurants
        migrate_restaurants(restaurants_csv, conn)
        
        # 2. Créer les familles de produits (corrigé)
        product_mapping = create_product_families(conn, transactions_csv)
        
        # 3. Migrer les transactions (corrigé)
        migrate_transactions(transactions_csv, conn, product_mapping)
        
        # 4. Créer les KPIs
        create_kpi_daily(conn)
        
        # 5. Créer des utilisateurs de test
        create_sample_users(conn)
        
        # 6. Créer des vues
        create_views(conn)
        
        # 7. Validation finale
        if validate_migration(conn):
            print("\n✅ Migration CORRIGÉE terminée avec succès!")
        else:
            print("\n⚠️  Migration terminée avec des avertissements")
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        
    finally:
        conn.close()

if __name__ == '__main__':
    main()