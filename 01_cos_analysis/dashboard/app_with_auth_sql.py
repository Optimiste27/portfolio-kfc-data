# dashboard/app_with_auth_sql.py
"""
Dashboard COS KFC COMPLET avec authentification et base SQL
Version migrée avec base de données SQLite - CORRIGÉE
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import base64
from pathlib import Path
import sqlite3

# === CONFIGURATION DES CHEMINS ===
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_dir = os.path.join(root_dir, 'src')
data_dir = os.path.join(root_dir, 'data', 'raw')
db_path = os.path.join(src_dir, 'cos_kfc.db')

sys.path.insert(0, src_dir)
sys.path.insert(0, root_dir)

# === AUTHENTIFICATION ===
try:
    from auth_simple import show_login, require_login, get_current_user, logout, can_access, get_user_role
    AUTH_OK = True
except ImportError:
    AUTH_OK = False
    st.error("❌ Module d'authentification manquant")

# Vérifier l'authentification
if not AUTH_OK:
    st.error("Système d'authentification indisponible")
    st.stop()

# Rediriger vers login si non connecté
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    show_login()
    st.stop()

# === DASHBOARD PRINCIPAL ===
user = get_current_user()
user_role = get_user_role()
user_name = user.get("name", user.get("username", "Utilisateur"))

# Configuration
st.set_page_config(
    page_title="Dashboard COS KFC",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    .pdf-section {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #007bff;
        margin: 1rem 0;
    }
    .user-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .kpi-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .success-badge { background-color: #d4edda; color: #155724; padding: 5px 10px; border-radius: 5px; }
    .warning-badge { background-color: #fff3cd; color: #856404; padding: 5px 10px; border-radius: 5px; }
    .danger-badge { background-color: #f8d7da; color: #721c24; padding: 5px 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# === CLASSE DATABASE POUR SQL ===
class Database:
    def __init__(self, db_path=db_path):
        self.db_path = db_path
    
    def get_connection(self):
        """Crée une connexion à la base SQLite"""
        return sqlite3.connect(self.db_path)
    
    def get_dataframe(self, query, params=None):
        """Retourne un DataFrame pandas depuis SQL"""
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
    
    def get_transactions(self, filters=None):
        """Récupère les transactions avec filtres"""
        base_query = '''
            SELECT 
                t.transaction_id,
                t.date,
                t.restaurant_id,
                r.restaurant_name,
                t.product_family_id,
                p.product_family_name,
                p.category as product_category,
                t.theoretical_unit_cost,
                t.actual_unit_cost,
                t.selling_price,
                t.units_sold,
                t.waste_kg,
                t.qsp_score,
                t.operational_anomaly,
                t.revenue,
                t.theoretical_cos,
                t.actual_cos,
                t.cos_gap,
                t.gap_percentage,
                t.gap_severity
            FROM transactions t
            JOIN restaurants r ON t.restaurant_id = r.restaurant_id
            JOIN product_families p ON t.product_family_id = p.product_family_id
            WHERE 1=1
        '''
        
        params = []
        
        if filters:
            if filters.get('start_date'):
                base_query += ' AND t.date >= ?'
                params.append(filters['start_date'])
            if filters.get('end_date'):
                base_query += ' AND t.date <= ?'
                params.append(filters['end_date'])
            if filters.get('restaurant_id'):
                base_query += ' AND t.restaurant_id = ?'
                params.append(filters['restaurant_id'])
            if filters.get('restaurant_name'):
                base_query += ' AND r.restaurant_name = ?'
                params.append(filters['restaurant_name'])
            if filters.get('product_family_id'):
                base_query += ' AND t.product_family_id = ?'
                params.append(filters['product_family_id'])
            if filters.get('product_family_name'):
                base_query += ' AND p.product_family_name = ?'
                params.append(filters['product_family_name'])
        
        base_query += ' ORDER BY t.date DESC, t.transaction_id DESC'
        
        df = self.get_dataframe(base_query, params)
        
        # Convertir les types de données
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'operational_anomaly' in df.columns:
            df['operational_anomaly'] = df['operational_anomaly'].astype(bool)
        
        return df
    
    def get_restaurants(self):
        """Récupère la liste des restaurants"""
        return self.get_dataframe('SELECT * FROM restaurants ORDER BY restaurant_name')
    
    def get_product_families(self):
        """Récupère la liste des produits"""
        return self.get_dataframe('SELECT * FROM product_families ORDER BY product_family_name')
    
    def get_kpi_daily(self):
        """Récupère les KPIs journaliers"""
        return self.get_dataframe('SELECT * FROM kpi_daily ORDER BY kpi_date DESC')

# Initialiser la base de données
db = Database()

# Import des modules PDF (si autorisé)
if can_access("manager"):  # Seuls manager+ peuvent générer des PDFs
    try:
        from pdf_reporter import COSPDFReporter
        PDF_AVAILABLE = True
        # Initialiser le générateur PDF
        reports_dir = os.path.join(root_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        pdf_reporter = COSPDFReporter(output_dir=reports_dir)
    except ImportError:
        PDF_AVAILABLE = False
else:
    PDF_AVAILABLE = False

# Sidebar avec profil utilisateur
with st.sidebar:
    # Badge utilisateur
    st.markdown(f'<div class="user-badge">👤 {user_name}<br><small>Rôle: {user_role.upper()}</small></div>', 
                unsafe_allow_html=True)
    
    # Bouton déconnexion
    if st.button("🚪 Déconnexion", use_container_width=True, type="secondary"):
        logout()
    
    st.markdown("---")
    st.header("📊 Navigation & Filtres")
    
    # Chargement des données depuis SQL
    if st.button("📥 Charger les données SQL", use_container_width=True, type="primary"):
        with st.spinner("Chargement depuis la base SQL..."):
            try:
                if os.path.exists(db_path):
                    # Charger les transactions
                    transactions_df = db.get_transactions()
                    
                    if transactions_df is not None and not transactions_df.empty:
                        st.session_state.transactions = transactions_df
                        st.session_state.data_loaded = True
                        st.success(f"✅ {len(transactions_df):,} transactions chargées")
                        st.rerun()
                    else:
                        st.error("❌ Aucune donnée trouvée dans la base SQL")
                else:
                    st.error(f"❌ Base de données non trouvée: {db_path}")
            except Exception as e:
                st.error(f"Erreur SQL: {e}")
    
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        transactions_df = st.session_state.transactions
        
        # Filtres (selon permissions)
        st.markdown("### 🔍 Filtres")
        
        # Filtre restaurant
        if 'restaurant_name' in transactions_df.columns:
            restaurants = ["Tous"] + sorted(transactions_df['restaurant_name'].unique().tolist())
            selected_restaurant = st.selectbox("🏪 Restaurant", restaurants)
        else:
            selected_restaurant = "Tous"
        
        # Filtre produit (manager+ seulement)
        if can_access("manager") and 'product_family_name' in transactions_df.columns:
            products = ["Toutes"] + sorted(transactions_df['product_family_name'].unique().tolist())
            selected_product = st.selectbox("🍗 Produit", products)
        else:
            selected_product = "Toutes"
            if 'product_family_name' not in transactions_df.columns:
                st.info("👁️ Mode consultation - Filtres limités")
        
        # Filtre date
        if 'date' in transactions_df.columns:
            min_date = transactions_df['date'].min().date()
            max_date = transactions_df['date'].max().date()
            date_range = st.date_input("📅 Période", [min_date, max_date])
        
        # Informations base SQL
        st.markdown("---")
        st.markdown("### 💾 Base SQL")
        
        try:
            # Statistiques de la base
            total_tx = len(transactions_df)
            unique_rest = transactions_df['restaurant_name'].nunique() if 'restaurant_name' in transactions_df.columns else 0
            unique_prod = transactions_df['product_family_name'].nunique() if 'product_family_name' in transactions_df.columns else 0
            
            db_size = os.path.getsize(db_path) / (1024*1024) if os.path.exists(db_path) else 0
            
            # Calculer quelques KPIs
            avg_gap = transactions_df['gap_percentage'].mean() if 'gap_percentage' in transactions_df.columns else 0
            total_revenue = transactions_df['revenue'].sum() if 'revenue' in transactions_df.columns else 0
            
            st.info(f"""
            **Base SQLite:**
            • 📊 {total_tx:,} transactions
            • 🏪 {unique_rest} restaurants
            • 🍗 {unique_prod} produits
            • 💰 €{total_revenue:,.0f} revenue
            • 📉 {avg_gap:.1f}% gap moyen
            • 💾 {db_size:.1f} MB
            • 👤 Rôle: {user_role.upper()}
            """)
        except Exception as e:
            st.error(f"Erreur stats: {str(e)[:50]}")

# Titre principal
st.title("🏪 Dashboard d'Analyse COS - KFC")
st.markdown(f"### Connecté en tant que **{user_name}** ({user_role}) • Base SQL • {datetime.now().strftime('%d/%m/%Y')}")

# Vérifier si les données sont chargées
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.info("👈 Veuillez charger les données depuis la sidebar")
    
    # Afficher des informations sur la migration
    with st.expander("ℹ️ Informations sur la migration SQL"):
        st.markdown("""
        ### 🗄️ Migration CSV → SQLite terminée !
        
        **Données migrées:**
        - ✅ 10 restaurants
        - ✅ 8 familles de produits
        - ✅ 14,400 transactions
        - ✅ 1,800 KPIs journaliers
        - ✅ Relations et contraintes SQL
        
        **Avantages:**
        - 🔒 Intégrité des données garantie
        - ⚡ Requêtes SQL optimisées
        - 📊 Analyses plus rapides
        - 🔄 Export multi-formats
        
        **Pour commencer:**
        1. Cliquez sur *"Charger les données SQL"* dans la sidebar
        2. Utilisez les filtres pour analyser les données
        3. Exportez en CSV/Excel si vous êtes manager
        """)
        
        # Vérifier si la base existe
        if os.path.exists(db_path):
            st.success(f"✅ Base de données trouvée: {db_path}")
            db_size = os.path.getsize(db_path) / (1024*1024)
            st.info(f"Taille: {db_size:.1f} MB")
        else:
            st.error(f"❌ Base de données non trouvée: {db_path}")
            st.markdown("""
            **Solution:**
            1. Exécutez le script de migration:
            ```bash
            cd src
            python csv_to_sql_migration_fixed.py
            ```
            2. Vérifiez que le fichier `cos_kfc.db` est créé
            """)
    st.stop()

# Récupérer les données
transactions_df = st.session_state.transactions

# Appliquer les filtres
filtered_data = transactions_df.copy()

if 'selected_restaurant' in locals() and selected_restaurant != "Tous":
    filtered_data = filtered_data[filtered_data['restaurant_name'] == selected_restaurant]

if 'selected_product' in locals() and selected_product != "Toutes" and can_access("manager"):
    filtered_data = filtered_data[filtered_data['product_family_name'] == selected_product]

if 'date_range' in locals() and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_data = filtered_data[
        (filtered_data['date'] >= pd.Timestamp(start_date)) & 
        (filtered_data['date'] <= pd.Timestamp(end_date))
    ]

# === ONGLETS ===
tab_names = ["📈 Vue d'ensemble", "🏪 Par restaurant", "🍗 Par produit", "📊 Détails"]
if PDF_AVAILABLE and can_access("manager"):
    tab_names.append("📄 Rapports PDF")

tabs = st.tabs(tab_names)

# Onglet 1: Vue d'ensemble
with tabs[0]:
    st.subheader("Vue d'ensemble des performances SQL")
    
    # Métriques KPI
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'gap_percentage' in filtered_data.columns:
            avg_gap = filtered_data['gap_percentage'].mean()
            delta = avg_gap - 4.0
            color = "normal" if abs(delta) < 1 else "inverse" if delta > 0 else "normal"
            st.metric("Écart COS moyen", f"{avg_gap:.2f}%", 
                     delta=f"{delta:+.2f}%", delta_color=color)
    
    with col2:
        if 'actual_unit_cost' in filtered_data.columns and 'units_sold' in filtered_data.columns:
            total_cost = (filtered_data['actual_unit_cost'] * filtered_data['units_sold']).sum()
            if 'theoretical_unit_cost' in filtered_data.columns:
                theoretical_cost = (filtered_data['theoretical_unit_cost'] * filtered_data['units_sold']).sum()
                cost_gap = total_cost - theoretical_cost
                st.metric("Surcoût total", f"{cost_gap:,.0f} €", 
                         delta_color="inverse" if cost_gap > 0 else "normal")
    
    with col3:
        if 'operational_anomaly' in filtered_data.columns:
            anomaly_rate = filtered_data['operational_anomaly'].mean() * 100
            st.metric("Taux d'anomalies", f"{anomaly_rate:.1f}%",
                     delta_color="inverse" if anomaly_rate > 5 else "normal")
    
    with col4:
        if 'waste_kg' in filtered_data.columns and 'theoretical_unit_cost' in filtered_data.columns:
            waste_cost = (filtered_data['waste_kg'] * filtered_data['theoretical_unit_cost']).sum()
            st.metric("Coût du gaspillage", f"{waste_cost:,.0f} €",
                     delta_color="inverse")
    
    # Graphiques avec Plotly
    col1, col2 = st.columns(2)
    
    with col1:
        if 'gap_percentage' in filtered_data.columns:
            fig1 = px.histogram(
                filtered_data, 
                x='gap_percentage',
                nbins=30,
                title='Distribution des écarts COS',
                labels={'gap_percentage': 'Écart COS (%)', 'count': 'Fréquence'},
                color_discrete_sequence=['skyblue']
            )
            fig1.add_vline(x=4.0, line_dash="dash", line_color="red", 
                          annotation_text="Cible (4.0%)")
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if 'restaurant_name' in filtered_data.columns and 'gap_percentage' in filtered_data.columns:
            restaurant_stats = filtered_data.groupby('restaurant_name')['gap_percentage'].mean().round(2)
            top5 = restaurant_stats.nlargest(5).reset_index()
            
            # Créer une couleur conditionnelle
            top5['color'] = top5['gap_percentage'].apply(
                lambda x: 'red' if x > 6 else 'orange' if x > 4 else 'green'
            )
            
            fig2 = px.bar(
                top5, 
                y='restaurant_name',
                x='gap_percentage',
                title='TOP 5 - Restaurants avec plus d\'écarts',
                labels={'gap_percentage': 'Écart COS moyen (%)', 'restaurant_name': 'Restaurant'},
                color='color',
                color_discrete_map={'red': 'red', 'orange': 'orange', 'green': 'green'}
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

# Onglet 2: Par restaurant (manager+ seulement)
with tabs[1]:
    if can_access("manager"):
        st.subheader("Analyse détaillée par restaurant")
        
        if 'restaurant_name' in filtered_data.columns:
            # Performance détaillée par restaurant
            agg_dict = {
                'gap_percentage': ['mean', 'std', 'count'],
                'revenue': 'sum',
                'units_sold': 'sum'
            }
            
            if 'operational_anomaly' in filtered_data.columns:
                agg_dict['operational_anomaly'] = 'mean'
            if 'qsp_score' in filtered_data.columns:
                agg_dict['qsp_score'] = 'mean'
            
            restaurant_perf = filtered_data.groupby('restaurant_name').agg(agg_dict).round(2)
            
            # Renommer les colonnes
            restaurant_perf.columns = ['_'.join(col).strip() for col in restaurant_perf.columns.values]
            
            # Créer un DataFrame propre
            display_df = pd.DataFrame({
                'Restaurant': restaurant_perf.index,
                'Écart moyen %': restaurant_perf.get('gap_percentage_mean', np.nan),
                'Std écart': restaurant_perf.get('gap_percentage_std', np.nan),
                'Nb transactions': restaurant_perf.get('gap_percentage_count', np.nan),
                'CA total (€)': restaurant_perf.get('revenue_sum', np.nan),
                'Unités vendues': restaurant_perf.get('units_sold_sum', np.nan),
            })
            
            if 'operational_anomaly_mean' in restaurant_perf.columns:
                display_df['Taux anomalies'] = restaurant_perf['operational_anomaly_mean'].round(3)
            if 'qsp_score_mean' in restaurant_perf.columns:
                display_df['Score QSP moyen'] = restaurant_perf['qsp_score_mean']
            
            # Ajouter un indicateur de performance
            display_df['Performance'] = display_df['Écart moyen %'].apply(
                lambda x: '🟢 Excellente' if x <= 3 else 
                         '🟡 Bonne' if x <= 5 else 
                         '🟠 Modérée' if x <= 7 else '🔴 Critique'
            )
            
            # Trier et afficher
            display_df = display_df.sort_values('Écart moyen %', ascending=False)
            
            # Afficher avec style conditionnel
            def color_performance(val):
                if val <= 3:
                    return 'background-color: #d4edda'  # Vert
                elif val <= 5:
                    return 'background-color: #fff3cd'  # Jaune
                elif val <= 7:
                    return 'background-color: #f8d7da'  # Rouge clair
                else:
                    return 'background-color: #dc3545; color: white'  # Rouge foncé
            
            styled_df = display_df.style.apply(
                lambda x: ['background-color: #d4edda' if v <= 3 else 
                          'background-color: #fff3cd' if v <= 5 else 
                          'background-color: #f8d7da' if v <= 7 else 
                          'background-color: #dc3545; color: white' 
                          for v in x] if x.name == 'Écart moyen %' else [''] * len(x)
            ).format({
                'Écart moyen %': '{:.2f}%',
                'Std écart': '{:.2f}',
                'CA total (€)': '€{:,.0f}',
                'Unités vendues': '{:,}',
                'Taux anomalies': '{:.1%}' if 'Taux anomalies' in display_df.columns else None,
                'Score QSP moyen': '{:.1f}' if 'Score QSP moyen' in display_df.columns else None
            })
            
            st.dataframe(
                styled_df,
                width='stretch',
                hide_index=True
            )
            
            # Télécharger les données
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporter cette analyse (CSV)",
                data=csv,
                file_name=f"analyse_restaurants_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("⛔ Accès restreint - Rôle 'manager' requis pour cette section")

# Onglet 3: Par produit (manager+ seulement)
with tabs[2]:
    if can_access("manager"):
        st.subheader("Analyse détaillée par produit")
        
        if 'product_family_name' in filtered_data.columns:
            # Performance par produit
            agg_dict = {
                'gap_percentage': ['mean', 'std'],
                'revenue': 'sum',
                'units_sold': 'sum',
                'operational_anomaly': 'mean'
            }
            
            if 'qsp_score' in filtered_data.columns:
                agg_dict['qsp_score'] = 'mean'
            
            product_perf = filtered_data.groupby('product_family_name').agg(agg_dict).round(2)
            product_perf.columns = ['_'.join(col).strip() for col in product_perf.columns.values]
            
            # Ajouter la catégorie si disponible
            if 'product_category' in filtered_data.columns:
                categories = filtered_data[['product_family_name', 'product_category']].drop_duplicates()
                product_perf = product_perf.merge(categories, left_index=True, right_on='product_family_name')
                product_perf.set_index('product_family_name', inplace=True)
            
            # Créer un DataFrame propre
            display_df = pd.DataFrame({
                'Produit': product_perf.index,
                'Catégorie': product_perf.get('product_category', 'N/A'),
                'Écart moyen %': product_perf.get('gap_percentage_mean', np.nan),
                'Std écart': product_perf.get('gap_percentage_std', np.nan),
                'CA total (€)': product_perf.get('revenue_sum', np.nan),
                'Unités vendues': product_perf.get('units_sold_sum', np.nan),
                'Taux anomalies': product_perf.get('operational_anomaly_mean', np.nan),
            })
            
            if 'qsp_score_mean' in product_perf.columns:
                display_df['Score QSP moyen'] = product_perf['qsp_score_mean']
            
            # Calculer le prix moyen
            display_df['Prix moyen (€)'] = (display_df['CA total (€)'] / display_df['Unités vendues']).round(2)
            
            # Trier par CA
            display_df = display_df.sort_values('CA total (€)', ascending=False)
            
            # Style conditionnel
            styled_df = display_df.style.apply(
                lambda x: ['background-color: #d4edda' if v <= 3 else 
                          'background-color: #fff3cd' if v <= 5 else 
                          'background-color: #f8d7da'
                          for v in x] if x.name == 'Écart moyen %' else [''] * len(x)
            ).format({
                'Écart moyen %': '{:.2f}%',
                'Std écart': '{:.2f}',
                'CA total (€)': '€{:,.0f}',
                'Unités vendues': '{:,}',
                'Taux anomalies': '{:.1%}',
                'Prix moyen (€)': '€{:.2f}',
                'Score QSP moyen': '{:.1f}' if 'Score QSP moyen' in display_df.columns else None
            })
            
            # Afficher
            st.dataframe(
                styled_df,
                width='stretch',
                hide_index=True
            )
            
            # Graphique des produits par CA
            fig = px.treemap(
                display_df,
                path=['Catégorie', 'Produit'],
                values='CA total (€)',
                color='Écart moyen %',
                color_continuous_scale='RdYlGn_r',
                title='Répartition du CA par produit (couleur = écart COS)',
                hover_data=['Unités vendues', 'Taux anomalies']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporter analyse produits (CSV)",
                data=csv,
                file_name=f"analyse_produits_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("⛔ Accès restreint - Rôle 'manager' requis pour cette section")

# Onglet 4: Détails (tous les rôles)
with tabs[3]:
    st.subheader("Détails des transactions SQL")
    
    # Colonnes à afficher selon le rôle
    display_cols = ['date', 'restaurant_name', 'gap_percentage', 'gap_severity']
    
    if can_access("manager"):
        display_cols.extend(['product_family_name', 'revenue', 'units_sold', 'operational_anomaly'])
    if can_access("admin"):
        display_cols.extend(['actual_unit_cost', 'theoretical_unit_cost', 'cos_gap'])
    
    # Limiter le nombre de lignes
    if user_role == "viewer":
        display_data = filtered_data[display_cols].head(100)  # Limite à 100 lignes
        st.info("👁️ Mode consultation - Affichage limité à 100 transactions")
    else:
        display_data = filtered_data[display_cols].head(500)  # 500 pour les managers
    
    # Formater les dates
    if 'date' in display_data.columns:
        display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d')
    
    # Style pour les anomalies - CORRECTION ICI
    if 'operational_anomaly' in display_data.columns:
        def highlight_anomalies(row):
            # Vérifier si c'est une anomalie (True/1)
            if row['operational_anomaly'] == 1 or row['operational_anomaly'] is True:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        styled_display = display_data.style.apply(highlight_anomalies, axis=1)
    else:
        styled_display = display_data.style
    
    # Formater les nombres - CORRECTION ICI
    format_dict = {}
    if 'gap_percentage' in display_data.columns:
        # gap_percentage est déjà un nombre, on le formate avec % à la fin
        styled_display = styled_display.format({
            'gap_percentage': '{:.2f}%'
        })
    
    if 'revenue' in display_data.columns:
        styled_display = styled_display.format({
            'revenue': '€{:,.2f}'
        })
    
    if 'actual_unit_cost' in display_data.columns:
        styled_display = styled_display.format({
            'actual_unit_cost': '€{:.2f}',
            'theoretical_unit_cost': '€{:.2f}'
        })
    
    st.dataframe(styled_display, width='stretch', height=400)
    
    # Statistiques - CORRECTION ICI (gap_percentage est déjà un nombre)
    st.markdown("#### 📊 Statistiques des transactions affichées")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nombre", f"{len(display_data):,}")
    
    with col2:
        if 'gap_percentage' in display_data.columns:
            # gap_percentage est déjà un float, pas besoin de .str.replace
            avg_gap = display_data['gap_percentage'].mean()
            st.metric("Écart moyen", f"{avg_gap:.2f}%")
    
    with col3:
        if 'operational_anomaly' in display_data.columns:
            # Convertir en bool/int si nécessaire
            if display_data['operational_anomaly'].dtype == object:
                anomalies = display_data['operational_anomaly'].apply(lambda x: 1 if str(x).lower() == 'true' else 0).sum()
            else:
                anomalies = display_data['operational_anomaly'].sum()
            st.metric("Anomalies", f"{anomalies}")
    
    # Export CSV (manager+ seulement)
    if can_access("manager"):
        csv = display_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter CSV",
            data=csv,
            file_name=f"cos_data_sql_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Onglet 5: Rapports PDF (manager+ seulement)
if len(tabs) > 4 and PDF_AVAILABLE and can_access("manager"):
    with tabs[4]:
        st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
        st.markdown("## 📄 Générateur de rapports PDF (SQL)")
        st.markdown("Générez des rapports professionnels à partir des données SQL")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Rapport hebdomadaire SQL", use_container_width=True, type="primary"):
                with st.spinner("Génération depuis SQL..."):
                    try:
                        # Charger les données restaurants depuis SQL
                        restaurants_df = db.get_restaurants()
                        
                        # Créer un simple rapport
                        report_text = f"""
                        Rapport COS KFC - Données SQL
                        Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                        
                        Statistiques globales:
                        - Total transactions: {len(filtered_data):,}
                        - Restaurants analysés: {filtered_data['restaurant_name'].nunique()}
                        - Produits analysés: {filtered_data['product_family_name'].nunique()}
                        - Écart COS moyen: {filtered_data['gap_percentage'].mean():.2f}%
                        - Revenue total: €{filtered_data['revenue'].sum():,.0f}
                        
                        Top 3 restaurants par écart:
                        """
                        
                        # Ajouter le top 3
                        top_restaurants = filtered_data.groupby('restaurant_name')['gap_percentage'].mean().nlargest(3)
                        for i, (restaurant, gap) in enumerate(top_restaurants.items(), 1):
                            report_text += f"\n{i}. {restaurant}: {gap:.2f}%"
                        
                        st.code(report_text, language='text')
                        st.success("Rapport généré (exemple - PDF à implémenter)")
                        
                    except Exception as e:
                        st.error(f"Erreur génération: {str(e)[:100]}")
        
        with col2:
            if 'restaurant_name' in filtered_data.columns:
                restaurant_options = filtered_data['restaurant_name'].unique()
                if len(restaurant_options) > 0:
                    selected_for_pdf = st.selectbox(
                        "Rapport restaurant SQL",
                        restaurant_options
                    )
                    if st.button(f"📋 Rapport {selected_for_pdf}", use_container_width=True):
                        with st.spinner("Génération..."):
                            st.info(f"Rapport pour {selected_for_pdf} - Fonctionnalité PDF à implémenter")
                else:
                    st.info("Aucun restaurant disponible")

# Footer avec info utilisateur et statut SQL
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <small>🏪 Dashboard COS KFC • <strong>Base SQL</strong> • Rôle: {user_role.upper()} • {user_name} • {datetime.now().strftime('%d/%m/%Y %H:%M')}</small>
    <br><small>🗄️ Migration CSV → SQLite terminée • {len(transactions_df):,} transactions • {transactions_df['restaurant_name'].nunique()} restaurants</small>
</div>
""", unsafe_allow_html=True)

# Bouton de rafraîchissement SQL
if st.sidebar.button("🔄 Rafraîchir depuis SQL", use_container_width=True):
    if 'transactions' in st.session_state:
        del st.session_state.transactions
    if 'data_loaded' in st.session_state:
        del st.session_state.data_loaded
    st.rerun()