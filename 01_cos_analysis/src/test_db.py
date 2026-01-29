"""
Test de la base de données SQLite
"""

import sqlite3
import pandas as pd

def test_database():
    """Teste la base de données"""
    print("🧪 TEST DE LA BASE DE DONNÉES COS KFC")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect('cos_kfc.db')
        
        # 1. Vérifier les tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        print("📊 TABLES DISPONIBLES:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   • {table_name:20} : {count:>6} enregistrements")
        
        print("\n🔍 Aperçu des données:")
        
        # 2. Aperçu restaurants
        print("\n🏪 RESTAURANTS (premiers 3):")
        df_restaurants = pd.read_sql_query("SELECT * FROM restaurants LIMIT 3", conn)
        print(df_restaurants.to_string(index=False))
        
        # 3. Aperçu transactions
        print("\n💳 TRANSACTIONS (premières 3):")
        df_transactions = pd.read_sql_query("SELECT date, restaurant_id, product_family_id, gap_percentage, revenue FROM transactions LIMIT 3", conn)
        print(df_transactions.to_string(index=False))
        
        # 4. Statistiques principales
        print("\n📈 STATISTIQUES PRINCIPALES:")
        
        # Nombre total de transactions
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_tx = cursor.fetchone()[0]
        print(f"   • Transactions totales : {total_tx:,}")
        
        # Écart moyen
        cursor.execute("SELECT AVG(gap_percentage) FROM transactions")
        avg_gap = cursor.fetchone()[0]
        print(f"   • Écart COS moyen : {avg_gap:.2f}%")
        
        # CA total
        cursor.execute("SELECT SUM(revenue) FROM transactions")
        total_revenue = cursor.fetchone()[0]
        print(f"   • Chiffre d'affaires total : {total_revenue:,.0f} €")
        
        # Top 3 restaurants par écart
        print("\n🎯 TOP 3 RESTAURANTS PAR ÉCART:")
        query_top3 = """
        SELECT 
            r.restaurant_name,
            ROUND(AVG(t.gap_percentage), 2) as avg_gap,
            COUNT(t.transaction_id) as nb_transactions,
            ROUND(SUM(t.revenue), 0) as total_revenue
        FROM restaurants r
        JOIN transactions t ON r.restaurant_id = t.restaurant_id
        GROUP BY r.restaurant_id, r.restaurant_name
        ORDER BY avg_gap DESC
        LIMIT 3
        """
        
        df_top3 = pd.read_sql_query(query_top3, conn)
        for _, row in df_top3.iterrows():
            print(f"   • {row['restaurant_name']}: {row['avg_gap']}% ({row['nb_transactions']} transactions, {row['total_revenue']:,.0f}€)")
        
        # 5. Test de performance
        print("\n⚡ TEST DE PERFORMANCE:")
        import time
        
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE gap_percentage > 6")
        critical_count = cursor.fetchone()[0]
        end_time = time.time()
        
        print(f"   • Transactions critiques (>6%): {critical_count}")
        print(f"   • Temps requête: {(end_time - start_time)*1000:.2f} ms")
        
        conn.close()
        
        print("\n" + "=" * 60)
        print("✅ TEST RÉUSSI - Base de données opérationnelle!")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        print("Vérifiez que la migration a bien fonctionné.")


if __name__ == "__main__":
    test_database()