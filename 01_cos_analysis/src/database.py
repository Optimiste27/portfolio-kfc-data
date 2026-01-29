# src/database.py (version mise à jour)
import sqlite3
import pandas as pd
from contextlib import contextmanager
from datetime import datetime

class Database:
    def __init__(self, db_path='cos_kfc.db'):
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """Contexte manager pour les connexions"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query, params=None, fetch=True):
        """Exécute une requête SQL"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                if 'SELECT' in query.upper() or 'PRAGMA' in query.upper():
                    return cursor.fetchall()
            
            conn.commit()
            return cursor.lastrowid if not fetch else None
    
    def get_dataframe(self, query, params=None):
        """Retourne un DataFrame pandas"""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    # Méthodes spécifiques au projet
    def get_restaurant_performance(self, restaurant_id=None, start_date=None, end_date=None):
        """Récupère les performances par restaurant"""
        query = '''
            SELECT * FROM v_daily_performance 
            WHERE 1=1
        '''
        params = []
        
        if restaurant_id:
            query += ' AND restaurant_id = ?'
            params.append(restaurant_id)
        if start_date:
            query += ' AND kpi_date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND kpi_date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY kpi_date'
        return self.get_dataframe(query, params)
    
    def get_product_analysis(self, product_family_id=None):
        """Analyse par produit"""
        query = '''
            SELECT 
                product_family_name,
                category,
                COUNT(*) as transaction_count,
                SUM(units_sold) as total_units,
                ROUND(SUM(revenue), 2) as total_revenue,
                ROUND(AVG(gap_percentage), 2) as avg_gap,
                ROUND(AVG(qsp_score), 1) as avg_qsp
            FROM v_transactions_detailed
            WHERE 1=1
        '''
        params = []
        
        if product_family_id:
            query += ' AND product_family_id = ?'
            params.append(product_family_id)
        
        query += ' GROUP BY product_family_name, category ORDER BY total_revenue DESC'
        return self.get_dataframe(query, params)
    
    def get_anomalies(self, limit=100):
        """Récupère les anomalies opérationnelles"""
        query = '''
            SELECT * FROM v_transactions_detailed 
            WHERE operational_anomaly = 1 
            ORDER BY date DESC 
            LIMIT ?
        '''
        return self.get_dataframe(query, [limit])