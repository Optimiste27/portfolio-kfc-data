# src/database_queries.py
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

class KFCDataManager:
    def __init__(self, db_path='cos_kfc.db'):
        self.db_path = db_path
    
    def get_connection(self):
        """Crée une connexion à la base"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
        return conn
    
    def get_restaurants(self):
        """Récupère tous les restaurants"""
        with self.get_connection() as conn:
            return pd.read_sql_query('SELECT * FROM restaurants ORDER BY restaurant_name', conn)
    
    def get_daily_performance(self, start_date=None, end_date=None):
        """Récupère les performances quotidiennes"""
        query = 'SELECT * FROM v_daily_performance WHERE 1=1'
        params = []
        
        if start_date:
            query += ' AND kpi_date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND kpi_date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY kpi_date, restaurant_id'
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_transactions(self, restaurant_id=None, product_family_id=None, date=None):
        """Récupère les transactions filtrées"""
        query = 'SELECT * FROM v_transactions_detailed WHERE 1=1'
        params = []
        
        if restaurant_id:
            query += ' AND restaurant_id = ?'
            params.append(restaurant_id)
        if product_family_id:
            query += ' AND product_family_id = ?'
            params.append(product_family_id)
        if date:
            query += ' AND date = ?'
            params.append(date)
        
        query += ' ORDER BY date DESC, transaction_id'
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_product_performance(self):
        """Performance par produit"""
        with self.get_connection() as conn:
            query = '''
                SELECT 
                    p.product_family_name,
                    p.category,
                    COUNT(t.transaction_id) as transaction_count,
                    SUM(t.units_sold) as total_units,
                    ROUND(SUM(t.revenue), 2) as total_revenue,
                    ROUND(AVG(t.gap_percentage), 2) as avg_gap_percentage,
                    ROUND(AVG(t.qsp_score), 2) as avg_qsp_score
                FROM transactions t
                JOIN product_families p ON t.product_family_id = p.product_family_id
                GROUP BY p.product_family_name, p.category
                ORDER BY total_revenue DESC
            '''
            return pd.read_sql_query(query, conn)
    
    def get_monthly_summary(self, year=2024):
        """Résumé mensuel"""
        with self.get_connection() as conn:
            query = '''
                SELECT 
                    strftime('%Y-%m', date) as month,
                    COUNT(*) as transaction_count,
                    SUM(units_sold) as total_units_sold,
                    ROUND(SUM(revenue), 2) as total_revenue,
                    ROUND(AVG(gap_percentage), 2) as avg_gap_percentage,
                    SUM(CASE WHEN operational_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count
                FROM transactions
                WHERE strftime('%Y', date) = ?
                GROUP BY strftime('%Y-%m', date)
                ORDER BY month
            '''
            return pd.read_sql_query(query, conn, params=(str(year),))

# Exemple d'utilisation
if __name__ == '__main__':
    db = KFCDataManager()
    
    print("=== TEST DES REQUÊTES ===")
    
    # 1. Restaurants
    restaurants = db.get_restaurants()
    print(f"Restaurants: {len(restaurants)}")
    print(restaurants[['restaurant_id', 'restaurant_name', 'region']].head())
    
    # 2. Performance quotidienne
    perf = db.get_daily_performance('2024-01-01', '2024-01-07')
    print(f"\nPerformance sur 7 jours: {len(perf)} enregistrements")
    
    # 3. Produits
    products = db.get_product_performance()
    print(f"\nPerformance par produit: {len(products)} produits")
    print(products.head())