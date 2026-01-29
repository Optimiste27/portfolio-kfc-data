# src/csv_to_sql_migration.py
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
    
    # Lire le CSV
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Extraire la région depuis le nom (ex: "KFC Paris Louvre" -> "Paris")
    df['region'] = df['restaurant_name'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else 'Inconnu')
    
    # Préparer les données pour la table restaurants
    restaurants_data = []
    for _, row in df.iterrows():
        restaurants_data.append((
            int(row['restaurant_id']),
            row['restaurant_name'],
            row['region'],
            '2020-01-01',  # opening_date par défaut
            f"Manager {row['restaurant_name']}",  # manager_name par défaut
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    # Insérer dans la base
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT OR REPLACE INTO restaurants 
        (restaurant_id, restaurant_name, region, opening_date, manager_name, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', restaurants_data)
    
    conn.commit()
    print(f"✓ {len(restaurants_data)} restaurants migrés")

def create_product_families(conn, transactions_csv_path):
    """Crée les familles de produits depuis les transactions"""
    print("Création des familles de produits...")
    
    # Lire les transactions pour extraire les familles uniques
    df = pd.read_csv(transactions_csv_path, encoding='utf-8')
    product_families = df['product_family'].unique()
    
    # Catégories par défaut basées sur le nom
    categories = {
        'Poulet': ['Poulet Frit', 'Tenders', 'Ailes de Poulet'],
        'Sandwich': ['Sandwichs', 'Burger Poulet', 'Wrap'],
        'Accompagnement': ['Frites', 'Potatoes', 'Salades'],
        'Dessert': ['Desserts', 'Boissons']
    }
    
    cursor = conn.cursor()
    
    # Vider la table existante
    cursor.execute('DELETE FROM product_families')
    
    # Insérer les familles
    for i, family in enumerate(product_families, 1):
        # Déterminer la catégorie
        category = 'Autre'
        for cat, families in categories.items():
            if any(fam in family for fam in families):
                category = cat
                break
        
        cursor.execute('''
            INSERT INTO product_families 
            (product_family_id, product_family_name, category, target_gap_percentage)
            VALUES (?, ?, ?, ?)
        ''', (i, family.strip(), category, 4.0))
    
    conn.commit()
    print(f"✓ {len(product_families)} familles de produits créées")
    return {family.strip(): i for i, family in enumerate(product_families, 1)}

def migrate_transactions(csv_path, conn, product_family_mapping):
    """Migre les transactions depuis CSV vers la base"""
    print("Migration des transactions...")
    
    # Lire le CSV
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Préparer les données
    transactions_data = []
    for _, row in df.iterrows():
        try:
            # Convertir product_family en ID
            product_family_id = product_family_mapping.get(row['product_family'].strip())
            if not product_family_id:
                print(f"Warning: Produit non trouvé: {row['product_family']}")
                continue
            
            # Nettoyer les valeurs booléennes
            operational_anomaly = 1 if str(row['operational_anomaly']).lower() == 'true' else 0
            
            transactions_data.append((
                int(row['transaction_id']),
                row['date'],
                int(row['restaurant_id']),
                product_family_id,
                float(row['theoretical_unit_cost']),
                float(row['actual_unit_cost']),
                float(row['selling_price']),
                int(row['units_sold']),
                float(row['waste_kg']) if pd.notna(row['waste_kg']) else 0.0,
                float(row['qsp_score']) if pd.notna(row['qsp_score']) else 0.0,
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
    
    # Insérer en batch
    cursor = conn.cursor()
    
    # Vérifier si on doit remplacer ou ignorer les doublons
    try:
        cursor.executemany('''
            INSERT OR REPLACE INTO transactions 
            (transaction_id, date, restaurant_id, product_family_id, 
             theoretical_unit_cost, actual_unit_cost, selling_price, units_sold,
             waste_kg, qsp_score, operational_anomaly, revenue,
             theoretical_cos, actual_cos, cos_gap, gap_percentage, gap_severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', transactions_data)
        
        conn.commit()
        print(f"✓ {len(transactions_data)} transactions migrées")
        
    except Exception as e:
        print(f"Erreur d'insertion: {e}")
        conn.rollback()

def create_kpi_daily(conn):
    """Calcule et insère les KPIs journaliers"""
    print("Création des KPIs journaliers...")
    
    cursor = conn.cursor()
    
    # Calculer les KPIs depuis les transactions
    cursor.execute('''
        INSERT OR REPLACE INTO kpi_daily
        (kpi_date, restaurant_id, total_transactions, avg_gap_percentage, 
         total_revenue, total_waste_cost, anomaly_rate)
        SELECT 
            date,
            restaurant_id,
            COUNT(*) as total_transactions,
            AVG(gap_percentage) as avg_gap_percentage,
            SUM(revenue) as total_revenue,
            SUM(waste_kg * theoretical_unit_cost) as total_waste_cost,
            AVG(CASE WHEN operational_anomaly = 1 THEN 1 ELSE 0 END) * 100 as anomaly_rate
        FROM transactions
        GROUP BY date, restaurant_id
        ORDER BY date, restaurant_id
    ''')
    
    conn.commit()
    
    # Vérifier le résultat
    cursor.execute('SELECT COUNT(*) FROM kpi_daily')
    count = cursor.fetchone()[0]
    print(f"✓ {count} enregistrements KPI créés")

def validate_migration(conn):
    """Valide l'intégrité des données migrées"""
    print("\n=== VALIDATION DE LA MIGRATION ===")
    
    cursor = conn.cursor()
    
    # 1. Vérifier le nombre d'enregistrements
    tables = ['restaurants', 'product_families', 'transactions', 'kpi_daily']
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {table}')
        count = cursor.fetchone()[0]
        print(f"{table}: {count} enregistrements")
    
    # 2. Vérifier les relations
    print("\nVérification des relations...")
    
    # Transactions sans restaurant correspondant
    cursor.execute('''
        SELECT COUNT(*) 
        FROM transactions t
        LEFT JOIN restaurants r ON t.restaurant_id = r.restaurant_id
        WHERE r.restaurant_id IS NULL
    ''')
    orphan_transactions = cursor.fetchone()[0]
    print(f"Transactions orphelines: {orphan_transactions}")
    
    # Transactions sans produit correspondant
    cursor.execute('''
        SELECT COUNT(*) 
        FROM transactions t
        LEFT JOIN product_families p ON t.product_family_id = p.product_family_id
        WHERE p.product_family_id IS NULL
    ''')
    orphan_products = cursor.fetchone()[0]
    print(f"Transactions avec produits invalides: {orphan_products}")
    
    # 3. Vérifier la cohérence des données
    print("\nVérification de la cohérence...")
    cursor.execute('SELECT MIN(date), MAX(date) FROM transactions')
    min_date, max_date = cursor.fetchone()
    print(f"Période couverte: {min_date} à {max_date}")
    
    cursor.execute('SELECT COUNT(DISTINCT restaurant_id) FROM transactions')
    unique_restaurants = cursor.fetchone()[0]
    print(f"Restaurants avec transactions: {unique_restaurants}")

def main():
    """Fonction principale de migration"""
    print("=== MIGRATION CSV → SQLite ===")
    
    # Chemins
    db_path = 'cos_kfc.db'
    restaurants_csv = '../data/raw/restaurants_final_20260124_1112.csv'
    transactions_csv = '../data/raw/transactions_final_20260124_1112.csv'
    
    # Connexion à la base
    conn = create_connection(db_path)
    
    try:
        # 1. Migrer les restaurants
        migrate_restaurants(restaurants_csv, conn)
        
        # 2. Créer les familles de produits
        product_mapping = create_product_families(conn, transactions_csv)
        
        # 3. Migrer les transactions
        migrate_transactions(transactions_csv, conn, product_mapping)
        
        # 4. Créer les KPIs journaliers
        create_kpi_daily(conn)
        
        # 5. Valider la migration
        validate_migration(conn)
        
        print("\n✅ Migration terminée avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la migration: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        
    finally:
        conn.close()

if __name__ == '__main__':
    main()