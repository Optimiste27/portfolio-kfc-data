# dashboard/dashboard_db.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import sqlite3

# Ajouter le chemin src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class Database:
    def __init__(self, db_path=None):
        if db_path is None:
            # Chemin relatif correct
            self.db_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'cos_kfc.db')
        else:
            self.db_path = db_path
    
    def get_connection(self):
        """Crée une connexion à la base"""
        conn = sqlite3.connect(self.db_path)
        return conn
    
    def get_dataframe(self, query, params=None):
        """Retourne un DataFrame pandas"""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def execute_query(self, query, params=None):
        """Exécute une requête SQL"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.fetchall()

class DashboardKFC:
    def __init__(self):
        # UTILISER NOTRE PROPRE CLASSE DATABASE
        self.db = Database()
        st.set_page_config(
            page_title="Dashboard COS KFC",
            page_icon="🍗",
            layout="wide"
        )
    
    def display_sidebar_filters(self):
        """Affiche les filtres dans la sidebar"""
        st.sidebar.title("📊 Filtres")
        
        # Vérifier d'abord si la base existe
        db_path = self.db.db_path
        if not os.path.exists(db_path):
            st.sidebar.error(f"Base de données non trouvée: {db_path}")
            return None
        
        # Dates
        min_date = datetime(2024, 1, 1).date()
        max_date = datetime(2024, 6, 28).date()
        
        date_range = st.sidebar.date_input(
            "Période",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Restaurants
        try:
            restaurants_df = self.db.get_dataframe(
                'SELECT restaurant_id, restaurant_name FROM restaurants ORDER BY restaurant_name'
            )
            restaurants = restaurants_df['restaurant_name'].tolist()
            selected_restaurants = st.sidebar.multiselect(
                "Restaurants",
                options=restaurants,
                default=restaurants[:3] if len(restaurants) > 3 else restaurants
            )
        except Exception as e:
            st.sidebar.error(f"Erreur chargement restaurants: {e}")
            selected_restaurants = []
        
        # Produits
        try:
            products_df = self.db.get_dataframe(
                'SELECT product_family_name FROM product_families ORDER BY product_family_name'
            )
            products = products_df['product_family_name'].tolist()
            selected_products = st.sidebar.multiselect(
                "Familles de produits",
                options=products,
                default=products[:3] if len(products) > 3 else products
            )
        except Exception as e:
            st.sidebar.error(f"Erreur chargement produits: {e}")
            selected_products = []
        
        return {
            'start_date': date_range[0] if len(date_range) > 0 else min_date,
            'end_date': date_range[1] if len(date_range) > 1 else max_date,
            'restaurants': selected_restaurants,
            'products': selected_products
        }
    
    def display_kpi_cards(self, filters):
        """Affiche les cartes KPI"""
        st.title("🍗 Dashboard COS KFC - Analyse des Coûts")
        
        # Vérification des filtres
        if filters is None:
            st.error("Impossible de charger les filtres. Vérifiez la base de données.")
            return
        
        # Récupérer les données filtrées
        query = '''
            SELECT 
                COUNT(DISTINCT restaurant_id) as restaurant_count,
                COUNT(*) as transaction_count,
                SUM(units_sold) as total_units,
                ROUND(SUM(revenue), 2) as total_revenue,
                ROUND(AVG(gap_percentage), 2) as avg_gap,
                SUM(CASE WHEN operational_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count
            FROM transactions t
            JOIN restaurants r ON t.restaurant_id = r.restaurant_id
            JOIN product_families p ON t.product_family_id = p.product_family_id
            WHERE date BETWEEN ? AND ?
        '''
        params = [filters['start_date'], filters['end_date']]
        
        # Ajouter les filtres restaurants
        if filters['restaurants']:
            placeholders = ','.join(['?'] * len(filters['restaurants']))
            query += f' AND r.restaurant_name IN ({placeholders})'
            params.extend(filters['restaurants'])
        
        # Ajouter les filtres produits
        if filters['products']:
            placeholders = ','.join(['?'] * len(filters['products']))
            query += f' AND p.product_family_name IN ({placeholders})'
            params.extend(filters['products'])
        
        try:
            kpis = self.db.get_dataframe(query, params)
            if kpis.empty:
                st.warning("Aucune donnée trouvée avec les filtres sélectionnés")
                return
            
            kpis = kpis.iloc[0]
            
            # Afficher les KPI
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🏪 Restaurants", kpis['restaurant_count'])
            
            with col2:
                st.metric("💰 Transactions", f"{kpis['transaction_count']:,}")
            
            with col3:
                st.metric("📦 Unités vendues", f"{kpis['total_units']:,}")
            
            with col4:
                st.metric("💶 Revenu total", f"€{kpis['total_revenue']:,.2f}")
            
            with col5:
                anomaly_rate = (kpis['anomaly_count'] / kpis['transaction_count'] * 100) if kpis['transaction_count'] > 0 else 0
                st.metric("⚠️ Anomalies", f"{anomaly_rate:.1f}%")
                
        except Exception as e:
            st.error(f"Erreur lors du calcul des KPIs: {e}")
    
    def display_performance_charts(self, filters):
        """Affiche les graphiques de performance"""
        if filters is None:
            return
        
        st.subheader("📈 Performance par restaurant")
        
        # Données pour le graphique
        query = '''
            SELECT 
                r.restaurant_name,
                r.region,
                ROUND(SUM(t.revenue), 2) as total_revenue,
                ROUND(AVG(t.gap_percentage), 2) as avg_gap_percentage,
                COUNT(*) as transaction_count
            FROM transactions t
            JOIN restaurants r ON t.restaurant_id = r.restaurant_id
            JOIN product_families p ON t.product_family_id = p.product_family_id
            WHERE t.date BETWEEN ? AND ?
        '''
        params = [filters['start_date'], filters['end_date']]
        
        if filters['restaurants']:
            placeholders = ','.join(['?'] * len(filters['restaurants']))
            query += f' AND r.restaurant_name IN ({placeholders})'
            params.extend(filters['restaurants'])
        
        if filters['products']:
            placeholders = ','.join(['?'] * len(filters['products']))
            query += f' AND p.product_family_name IN ({placeholders})'
            params.extend(filters['products'])
        
        query += ' GROUP BY r.restaurant_id, r.restaurant_name, r.region ORDER BY total_revenue DESC'
        
        try:
            performance_df = self.db.get_dataframe(query, params)
            
            if performance_df.empty:
                st.info("Aucune donnée à afficher")
                return
            
            # Graphique 1: Revenu par restaurant
            fig1 = px.bar(
                performance_df,
                x='restaurant_name',
                y='total_revenue',
                title="Revenu total par restaurant",
                color='avg_gap_percentage',
                color_continuous_scale='RdYlGn_r',
                labels={'total_revenue': 'Revenu (€)', 'restaurant_name': 'Restaurant', 'avg_gap_percentage': 'Gap moyen %'},
                hover_data=['region', 'transaction_count']
            )
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, width='stretch')
            
            # Graphique 2: Évolution temporelle du gap
            st.subheader("📅 Évolution du COS Gap dans le temps")
            
            time_query = '''
                SELECT 
                    t.date,
                    ROUND(AVG(t.gap_percentage), 2) as avg_daily_gap,
                    ROUND(SUM(t.revenue), 2) as daily_revenue
                FROM transactions t
                JOIN restaurants r ON t.restaurant_id = r.restaurant_id
                JOIN product_families p ON t.product_family_id = p.product_family_id
                WHERE t.date BETWEEN ? AND ?
            '''
            time_params = [filters['start_date'], filters['end_date']]
            
            if filters['restaurants']:
                placeholders = ','.join(['?'] * len(filters['restaurants']))
                time_query += f' AND r.restaurant_name IN ({placeholders})'
                time_params.extend(filters['restaurants'])
            
            time_query += ' GROUP BY t.date ORDER BY t.date'
            
            time_df = self.db.get_dataframe(time_query, time_params)
            
            if not time_df.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=time_df['date'], 
                    y=time_df['avg_daily_gap'],
                    mode='lines+markers',
                    name='COS Gap moyen',
                    line=dict(color='red', width=2)
                ))
                fig2.add_trace(go.Bar(
                    x=time_df['date'],
                    y=time_df['daily_revenue'],
                    name='Revenu quotidien',
                    yaxis='y2',
                    opacity=0.3
                ))
                
                fig2.update_layout(
                    title="Évolution du COS Gap et du revenu",
                    yaxis=dict(title="COS Gap moyen (%)"),
                    yaxis2=dict(title="Revenu (€)", overlaying='y', side='right'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig2, width='stretch')
            else:
                st.info("Pas assez de données temporelles pour afficher le graphique")
                
        except Exception as e:
            st.error(f"Erreur lors de la création des graphiques: {e}")
    
    def display_product_analysis(self, filters):
        """Analyse par produit"""
        if filters is None:
            return
        
        st.subheader("🍔 Analyse par famille de produits")
        
        query = '''
            SELECT 
                p.product_family_name,
                p.category,
                COUNT(*) as transaction_count,
                SUM(t.units_sold) as total_units,
                ROUND(SUM(t.revenue), 2) as total_revenue,
                ROUND(AVG(t.gap_percentage), 2) as avg_gap,
                ROUND(AVG(t.qsp_score), 1) as avg_qsp
            FROM transactions t
            JOIN product_families p ON t.product_family_id = p.product_family_id
            JOIN restaurants r ON t.restaurant_id = r.restaurant_id
            WHERE t.date BETWEEN ? AND ?
        '''
        params = [filters['start_date'], filters['end_date']]
        
        if filters['restaurants']:
            placeholders = ','.join(['?'] * len(filters['restaurants']))
            query += f' AND r.restaurant_name IN ({placeholders})'
            params.extend(filters['restaurants'])
        
        if filters['products']:
            placeholders = ','.join(['?'] * len(filters['products']))
            query += f' AND p.product_family_name IN ({placeholders})'
            params.extend(filters['products'])
        
        query += ' GROUP BY p.product_family_name, p.category ORDER BY total_revenue DESC'
        
        try:
            product_df = self.db.get_dataframe(query, params)
            
            if product_df.empty:
                st.info("Aucune donnée produit à afficher")
                return
            
            # Graphique treemap
            fig = px.treemap(
                product_df,
                path=['category', 'product_family_name'],
                values='total_revenue',
                color='avg_gap',
                color_continuous_scale='RdYlGn_r',
                title="Répartition du revenu par produit (couleur = COS Gap)",
                hover_data=['total_units', 'avg_qsp', 'transaction_count']
            )
            st.plotly_chart(fig, width='stretch')
            
            # Tableau détaillé
            st.subheader("📋 Détail par produit")
            
            # Créer un style pour le dataframe
            styled_df = product_df.style.format({
                'total_revenue': '€{:,.2f}',
                'avg_gap': '{:.2f}%',
                'avg_qsp': '{:.1f}',
                'total_units': '{:,}',
                'transaction_count': '{:,}'
            })
            
            # Appliquer le gradient de couleur pour le gap
            styled_df = styled_df.background_gradient(
                subset=['avg_gap'], 
                cmap='RdYlGn_r'
            )
            
            st.dataframe(
                styled_df,
                width='stretch',
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse des produits: {e}")
    
    def display_daily_anomalies(self, filters):
        """Affiche les anomalies quotidiennes"""
        if filters is None:
            return
        
        with st.expander("⚠️ Anomalies opérationnelles détaillées"):
            try:
                anomalies_query = '''
                    SELECT 
                        t.date,
                        r.restaurant_name,
                        p.product_family_name,
                        t.units_sold,
                        t.revenue,
                        t.gap_percentage,
                        t.gap_severity,
                        t.operational_anomaly
                    FROM transactions t
                    JOIN restaurants r ON t.restaurant_id = r.restaurant_id
                    JOIN product_families p ON t.product_family_id = p.product_family_id
                    WHERE t.date BETWEEN ? AND ?
                    AND t.operational_anomaly = 1
                    ORDER BY t.gap_percentage DESC
                    LIMIT 50
                '''
                
                anomalies_data = self.db.get_dataframe(anomalies_query, 
                    [filters['start_date'], filters['end_date']])
                
                if not anomalies_data.empty:
                    # Créer un style pour les anomalies
                    def highlight_anomalies(row):
                        if row['operational_anomaly'] == 1:
                            return ['background-color: #ffcccc'] * len(row)
                        return [''] * len(row)
                    
                    styled_anomalies = anomalies_data.style.format({
                        'revenue': '€{:,.2f}',
                        'gap_percentage': '{:.2f}%'
                    }).apply(highlight_anomalies, axis=1)
                    
                    st.dataframe(
                        styled_anomalies,
                        width='stretch',
                        hide_index=True
                    )
                    
                    # Statistiques des anomalies
                    st.subheader("📊 Statistiques des anomalies")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_anomalies = len(anomalies_data)
                        st.metric("Total anomalies", total_anomalies)
                    
                    with col2:
                        avg_gap_anomalies = anomalies_data['gap_percentage'].mean()
                        st.metric("Gap moyen anomalies", f"{avg_gap_anomalies:.2f}%")
                    
                    with col3:
                        total_loss = (anomalies_data['revenue'] * (anomalies_data['gap_percentage']/100)).sum()
                        st.metric("Perte estimée", f"€{total_loss:.2f}")
                        
                else:
                    st.success("✅ Aucune anomalie détectée sur la période sélectionnée")
                    
            except Exception as e:
                st.error(f"Erreur chargement anomalies: {e}")
    
    def display_database_info(self):
        """Affiche des informations sur la base de données"""
        with st.sidebar.expander("ℹ️ Info Base de données"):
            db_path = self.db.db_path
            exists = os.path.exists(db_path)
            
            st.write(f"**Chemin:** `{db_path}`")
            st.write(f"**Existe:** {'✅ Oui' if exists else '❌ Non'}")
            
            if exists:
                size_bytes = os.path.getsize(db_path)
                size_mb = size_bytes / 1024 / 1024
                st.write(f"**Taille:** {size_mb:.2f} MB")
                
                # Nombre d'enregistrements
                try:
                    counts = {}
                    tables = ['restaurants', 'product_families', 'transactions', 'kpi_daily']
                    for table in tables:
                        count_df = self.db.get_dataframe(f'SELECT COUNT(*) as cnt FROM {table}')
                        counts[table] = count_df.iloc[0]['cnt']
                    
                    st.write("**Enregistrements:**")
                    for table, count in counts.items():
                        st.write(f"  - {table}: {count:,}")
                        
                    # Dernière mise à jour
                    try:
                        last_update = self.db.get_dataframe(
                            "SELECT MAX(date) as last_date FROM transactions"
                        ).iloc[0]['last_date']
                        st.write(f"**Dernière mise à jour:** {last_update}")
                    except:
                        pass
                        
                except Exception as e:
                    st.write(f"**Erreur lecture:** {str(e)[:50]}...")
    
    def display_quick_stats(self):
        """Affiche des statistiques rapides dans la sidebar"""
        with st.sidebar.expander("📈 Stats rapides"):
            try:
                # Revenu total
                revenue = self.db.get_dataframe(
                    "SELECT ROUND(SUM(revenue), 2) as total FROM transactions"
                ).iloc[0]['total']
                st.metric("Revenu total", f"€{revenue:,.2f}")
                
                # Transactions totales
                transactions = self.db.get_dataframe(
                    "SELECT COUNT(*) as total FROM transactions"
                ).iloc[0]['total']
                st.metric("Transactions", f"{transactions:,}")
                
                # Restaurants actifs
                restaurants = self.db.get_dataframe(
                    "SELECT COUNT(DISTINCT restaurant_id) as total FROM transactions"
                ).iloc[0]['total']
                st.metric("Restaurants actifs", restaurants)
                
            except Exception as e:
                st.error(f"Erreur stats: {str(e)[:50]}")
    
    def display_export_options(self):
        """Options d'export dans la sidebar"""
        with st.sidebar.expander("💾 Exporter les données"):
            st.write("Exportez les données au format:")
            
            if st.button("📄 CSV (toutes les tables)"):
                self.export_data('csv')
            
            if st.button("📊 Excel (rapport complet)"):
                self.export_data('excel')
            
            if st.button("📋 JSON (pour API)"):
                self.export_data('json')
    
    def export_data(self, format_type):
        """Exporte les données"""
        try:
            import tempfile
            import io
            
            # Liste des tables
            tables = {
                'restaurants': 'Restaurants',
                'product_families': 'Familles de produits',
                'transactions': 'Transactions',
                'kpi_daily': 'KPIs quotidiens'
            }
            
            if format_type == 'csv':
                # Créer un fichier ZIP avec tous les CSV
                import zipfile
                
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w') as zip_file:
                    for table, name in tables.items():
                        df = self.db.get_dataframe(f'SELECT * FROM {table}')
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        zip_file.writestr(f'{name}.csv', csv_buffer.getvalue())
                
                buffer.seek(0)
                st.download_button(
                    label="📥 Télécharger ZIP",
                    data=buffer,
                    file_name=f"cos_kfc_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
                
            elif format_type == 'excel':
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    for table, name in tables.items():
                        df = self.db.get_dataframe(f'SELECT * FROM {table}')
                        df.to_excel(writer, sheet_name=name[:31], index=False)
                
                buffer.seek(0)
                st.download_button(
                    label="📥 Télécharger Excel",
                    data=buffer,
                    file_name=f"cos_kfc_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            elif format_type == 'json':
                all_data = {}
                for table in tables.keys():
                    df = self.db.get_dataframe(f'SELECT * FROM {table}')
                    all_data[table] = df.to_dict(orient='records')
                
                import json
                json_str = json.dumps(all_data, indent=2, ensure_ascii=False, default=str)
                
                st.download_button(
                    label="📥 Télécharger JSON",
                    data=json_str,
                    file_name=f"cos_kfc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"Erreur lors de l'export: {e}")
    
    def run(self):
        """Exécute le dashboard"""
        # Afficher les infos DB dans la sidebar
        self.display_database_info()
        self.display_quick_stats()
        self.display_export_options()
        
        # Récupérer les filtres
        filters = self.display_sidebar_filters()
        
        # Si pas de filtres (erreur DB), afficher message
        if filters is None:
            st.error("""
            ## ⚠️ Base de données non trouvée
            
            La base de données `cos_kfc.db` n'a pas été trouvée dans `../src/`.
            
            **Solutions:**
            1. Vérifiez que la migration a été exécutée: `python src/csv_to_sql_migration_fixed.py`
            2. Vérifiez le chemin: `../src/cos_kfc.db`
            3. Exécutez le dashboard console: `python dashboard/console_dashboard.py`
            """)
            return
        
        # Afficher le contenu
        self.display_kpi_cards(filters)
        self.display_performance_charts(filters)
        self.display_product_analysis(filters)
        self.display_daily_anomalies(filters)
        
        # Section détaillée des transactions
        with st.expander("🔍 Voir les transactions détaillées"):
            try:
                sample_query = '''
                    SELECT 
                        t.date,
                        r.restaurant_name,
                        p.product_family_name,
                        t.units_sold,
                        t.revenue,
                        t.gap_percentage,
                        t.gap_severity,
                        t.operational_anomaly
                    FROM transactions t
                    JOIN restaurants r ON t.restaurant_id = r.restaurant_id
                    JOIN product_families p ON t.product_family_id = p.product_family_id
                    WHERE t.date BETWEEN ? AND ?
                    ORDER BY t.date DESC
                    LIMIT 100
                '''
                
                sample_data = self.db.get_dataframe(sample_query, 
                    [filters['start_date'], filters['end_date']])
                
                if not sample_data.empty:
                    # Fonction de style pour les anomalies
                    def highlight_rows(row):
                        if row['operational_anomaly'] == 1:
                            return ['background-color: #ffcccc'] * len(row)
                        return [''] * len(row)
                    
                    styled_data = sample_data.style.format({
                        'revenue': '€{:,.2f}',
                        'gap_percentage': '{:.2f}%'
                    }).apply(highlight_rows, axis=1)
                    
                    st.dataframe(
                        styled_data,
                        width='stretch',
                        hide_index=True
                    )
                else:
                    st.info("Aucune transaction trouvée")
            except Exception as e:
                st.error(f"Erreur chargement transactions: {e}")
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Dashboard COS KFC")
        st.sidebar.markdown(f"**Version:** 1.0 • **Données:** {datetime.now().strftime('%d/%m/%Y')}")
        st.sidebar.markdown("Migration CSV → SQLite complétée ✅")

if __name__ == "__main__":
    dashboard = DashboardKFC()
    dashboard.run()