"""
Dashboard COS KFC COMPLET avec authentification
Version authentifiée de app.py
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

# === CONFIGURATION DES CHEMINS ===
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_dir = os.path.join(root_dir, 'src')
data_dir = os.path.join(root_dir, 'data', 'raw')

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
</style>
""", unsafe_allow_html=True)

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
    
    # Chargement des données (disponible pour tous les rôles)
    @st.cache_data
    def load_data():
        """Charge les données depuis data/raw"""
        try:
            # Cherche les fichiers les plus récents
            if os.path.exists(data_dir):
                csv_files = list(Path(data_dir).glob("*.csv"))
                trans_files = [f for f in csv_files if 'transaction' in f.name.lower()]
                
                if trans_files:
                    latest = max(trans_files, key=lambda x: x.stat().st_mtime)
                    transactions = pd.read_csv(latest)
                    
                    # Conversion de date
                    if 'date' in transactions.columns:
                        transactions['date'] = pd.to_datetime(transactions['date'])
                    
                    return transactions
        except Exception as e:
            st.error(f"Erreur: {e}")
        return None
    
    # Charger les données
    data_loaded = False
    if st.button("📥 Charger les données", use_container_width=True, type="primary"):
        with st.spinner("Chargement..."):
            transactions_df = load_data()
            if transactions_df is not None:
                st.session_state.transactions = transactions_df
                st.session_state.data_loaded = True
                st.success(f"✅ {len(transactions_df):,} transactions chargées")
                st.rerun()
            else:
                st.error("❌ Erreur de chargement")
    
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
        if can_access("manager") and 'product_family' in transactions_df.columns:
            products = ["Toutes"] + sorted(transactions_df['product_family'].unique().tolist())
            selected_product = st.selectbox("🍗 Produit", products)
        else:
            selected_product = "Toutes"
            if 'product_family' not in transactions_df.columns:
                st.info("👁️ Mode consultation - Filtres limités")
        
        # Filtre date
        if 'date' in transactions_df.columns:
            min_date = transactions_df['date'].min().date()
            max_date = transactions_df['date'].max().date()
            date_range = st.date_input("📅 Période", [min_date, max_date])
        
        # Informations
        st.markdown("---")
        st.markdown("### 📊 Informations")
        
        total_tx = len(transactions_df)
        unique_rest = transactions_df['restaurant_name'].nunique() if 'restaurant_name' in transactions_df.columns else 0
        
        st.info(f"""
        **Données chargées:**
        • {total_tx:,} transactions
        • {unique_rest} restaurants
        • Rôle: {user_role.upper()}
        • PDF: {'✅' if PDF_AVAILABLE else '❌'}
        """)

# Titre principal
st.title("🏪 Dashboard d'Analyse COS - KFC")
st.markdown(f"### Connecté en tant que **{user_name}** ({user_role}) • Surveillance des écarts de coût opérationnel")

# Vérifier si les données sont chargées
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.info("👈 Veuillez charger les données depuis la sidebar")
    st.stop()

# Récupérer les données
transactions_df = st.session_state.transactions

# Appliquer les filtres
filtered_data = transactions_df.copy()

if 'selected_restaurant' in locals() and selected_restaurant != "Tous":
    filtered_data = filtered_data[filtered_data['restaurant_name'] == selected_restaurant]

if 'selected_product' in locals() and selected_product != "Toutes" and can_access("manager"):
    filtered_data = filtered_data[filtered_data['product_family'] == selected_product]

if 'date_range' in locals() and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_data = filtered_data[
        (filtered_data['date'] >= pd.Timestamp(start_date)) & 
        (filtered_data['date'] <= pd.Timestamp(end_date))
    ]

# === ONGLETS (identique à app.py original) ===
tab_names = ["📈 Vue d'ensemble", "🏪 Par restaurant", "🍗 Par produit", "📊 Détails"]
if PDF_AVAILABLE and can_access("manager"):
    tab_names.append("📄 Rapports PDF")

tabs = st.tabs(tab_names)

# Onglet 1: Vue d'ensemble (identique à app.py)
with tabs[0]:
    st.subheader("Vue d'ensemble des performances")
    
    # Métriques KPI
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'gap_percentage' in filtered_data.columns:
            avg_gap = filtered_data['gap_percentage'].mean()
            st.metric("Écart COS moyen", f"{avg_gap:.2f}%", delta=f"{(avg_gap - 4.0):.2f}%")
    
    with col2:
        if 'actual_unit_cost' in filtered_data.columns and 'units_sold' in filtered_data.columns:
            total_cost = (filtered_data['actual_unit_cost'] * filtered_data['units_sold']).sum()
            if 'theoretical_unit_cost' in filtered_data.columns:
                theoretical_cost = (filtered_data['theoretical_unit_cost'] * filtered_data['units_sold']).sum()
                cost_gap = total_cost - theoretical_cost
                st.metric("Surcoût total", f"{cost_gap:,.0f} €")
    
    with col3:
        if 'operational_anomaly' in filtered_data.columns:
            anomaly_rate = filtered_data['operational_anomaly'].mean() * 100
            st.metric("Taux d'anomalies", f"{anomaly_rate:.1f}%")
    
    with col4:
        if 'waste_kg' in filtered_data.columns and 'theoretical_unit_cost' in filtered_data.columns:
            waste_cost = (filtered_data['waste_kg'] * filtered_data['theoretical_unit_cost']).sum()
            st.metric("Coût du gaspillage", f"{waste_cost:,.0f} €")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        if 'gap_percentage' in filtered_data.columns:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.hist(filtered_data['gap_percentage'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(4.0, color='red', linestyle='--', label='Cible (4.0%)')
            ax1.set_xlabel('Écart COS (%)')
            ax1.set_ylabel('Fréquence')
            ax1.set_title('Distribution des écarts COS')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
    
    with col2:
        if 'restaurant_name' in filtered_data.columns and 'gap_percentage' in filtered_data.columns:
            restaurant_stats = filtered_data.groupby('restaurant_name')['gap_percentage'].mean().round(2)
            top5 = restaurant_stats.nlargest(5)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            colors = ['red' if x > 6 else 'orange' for x in top5.values]
            ax2.barh(top5.index, top5.values, color=colors, edgecolor='black')
            ax2.set_xlabel('Écart COS moyen (%)')
            ax2.set_title('TOP 5 - Restaurants avec plus d\'écarts')
            for i, (idx, val) in enumerate(top5.items()):
                ax2.text(val + 0.1, i, f"{val}%", va='center')
            st.pyplot(fig2)

# Onglet 2: Par restaurant (manager+ seulement)
with tabs[1]:
    if can_access("manager"):
        st.subheader("Analyse par restaurant")
        
        if 'restaurant_name' in filtered_data.columns:
            # Performance détaillée par restaurant
            agg_dict = {'gap_percentage': ['mean', 'std', 'count']}
            if 'operational_anomaly' in filtered_data.columns:
                agg_dict['operational_anomaly'] = 'mean'
            if 'revenue' in filtered_data.columns:
                agg_dict['revenue'] = 'sum'
            
            restaurant_perf = filtered_data.groupby('restaurant_name').agg(agg_dict).round(2)
            
            # Renommer les colonnes
            restaurant_perf.columns = ['_'.join(col).strip() for col in restaurant_perf.columns.values]
            restaurant_perf = restaurant_perf.rename(columns={
                'gap_percentage_mean': 'Écart moyen %',
                'gap_percentage_std': 'Std écart',
                'gap_percentage_count': 'Nb transactions'
            })
            
            if 'operational_anomaly_mean' in restaurant_perf.columns:
                restaurant_perf = restaurant_perf.rename(columns={'operational_anomaly_mean': 'Taux anomalies'})
            if 'revenue_sum' in restaurant_perf.columns:
                restaurant_perf = restaurant_perf.rename(columns={'revenue_sum': 'CA total'})
            
            st.dataframe(
                restaurant_perf.sort_values('Écart moyen %', ascending=False),
                width='stretch'
            )
    else:
        st.warning("⛔ Accès restreint - Rôle 'manager' requis pour cette section")

# Onglet 3: Par produit (manager+ seulement)
with tabs[2]:
    if can_access("manager"):
        st.subheader("Analyse par produit")
        
        if 'product_family' in filtered_data.columns:
            # Performance par produit
            agg_dict = {'gap_percentage': ['mean', 'std']}
            if 'revenue' in filtered_data.columns:
                agg_dict['revenue'] = 'sum'
            if 'units_sold' in filtered_data.columns:
                agg_dict['units_sold'] = 'sum'
            
            product_perf = filtered_data.groupby('product_family').agg(agg_dict).round(2)
            
            # Renommer
            product_perf.columns = ['_'.join(col).strip() for col in product_perf.columns.values]
            rename_map = {'gap_percentage_mean': 'Écart moyen %', 'gap_percentage_std': 'Std écart'}
            if 'revenue_sum' in product_perf.columns:
                rename_map['revenue_sum'] = 'CA total'
            if 'units_sold_sum' in product_perf.columns:
                rename_map['units_sold_sum'] = 'Unités vendues'
            
            product_perf = product_perf.rename(columns=rename_map)
            
            st.dataframe(
                product_perf.sort_values('Écart moyen %', ascending=False),
                width='stretch'
            )
    else:
        st.warning("⛔ Accès restreint - Rôle 'manager' requis pour cette section")

# Onglet 4: Détails (tous les rôles)
with tabs[3]:
    st.subheader("Détails des transactions")
    
    # Colonnes à afficher selon le rôle
    display_cols = ['date', 'restaurant_name']
    if can_access("manager"):
        display_cols.extend(['product_family', 'gap_percentage', 'revenue'])
    else:
        display_cols.extend(['gap_percentage'])  # Viewer voit moins de colonnes
    
    if 'gap_severity' in filtered_data.columns:
        display_cols.append('gap_severity')
    
    # Limiter le nombre de lignes pour les viewers
    if user_role == "viewer":
        display_data = filtered_data[display_cols].head(100)  # Limite à 100 lignes
        st.info("👁️ Mode consultation - Affichage limité à 100 transactions")
    else:
        display_data = filtered_data[display_cols]
    
    st.dataframe(display_data, width='stretch', height=400)
    
    # Export CSV (manager+ seulement)
    if can_access("manager"):
        csv = display_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter CSV",
            data=csv,
            file_name=f"cos_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Onglet 5: Rapports PDF (manager+ seulement)
if len(tabs) > 4 and PDF_AVAILABLE and can_access("manager"):
    with tabs[4]:
        st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
        st.markdown("## 📄 Générateur de rapports PDF")
        st.markdown("Générez des rapports professionnels en PDF")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Rapport hebdomadaire", use_container_width=True, type="primary"):
                with st.spinner("Génération..."):
                    try:
                        # Charger les infos restaurants
                        rest_files = list(Path(data_dir).glob("*restaurant*.csv"))
                        if rest_files:
                            restaurants_df = pd.read_csv(max(rest_files, key=lambda x: x.stat().st_mtime))
                        else:
                            restaurants_df = pd.DataFrame()
                        
                        report_path = pdf_reporter.generate_weekly_report(
                            filtered_data,
                            restaurants_df,
                            week_number=1
                        )
                        
                        if report_path:
                            st.success("✅ Rapport généré!")
                            # Lien de téléchargement
                            with open(report_path, "rb") as f:
                                pdf_bytes = f.read()
                            b64 = base64.b64encode(pdf_bytes).decode()
                            href = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(report_path)}" style="background:#007bff;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;font-weight:bold;">📥 Télécharger le PDF</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erreur: {str(e)[:100]}")
        
        with col2:
            if 'restaurant_name' in filtered_data.columns:
                selected_for_pdf = st.selectbox(
                    "Rapport restaurant",
                    filtered_data['restaurant_name'].unique()
                )
                if st.button(f"📋 Rapport {selected_for_pdf}", use_container_width=True):
                    with st.spinner("Génération..."):
                        try:
                            rest_files = list(Path(data_dir).glob("*restaurant*.csv"))
                            if rest_files:
                                restaurants_df = pd.read_csv(max(rest_files, key=lambda x: x.stat().st_mtime))
                            else:
                                restaurants_df = pd.DataFrame()
                            
                            report_path = pdf_reporter.generate_restaurant_report(
                                filtered_data,
                                restaurants_df,
                                selected_for_pdf
                            )
                            
                            if report_path:
                                st.success("✅ Rapport généré!")
                                with open(report_path, "rb") as f:
                                    pdf_bytes = f.read()
                                b64 = base64.b64encode(pdf_bytes).decode()
                                href = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(report_path)}" style="background:#28a745;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;font-weight:bold;">📥 Télécharger</a>'
                                st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erreur: {str(e)[:100]}")

# Footer avec info utilisateur
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <small>🏪 Dashboard COS KFC • Rôle: {user_role.upper()} • {user_name} • {datetime.now().strftime('%d/%m/%Y %H:%M')}</small>
</div>
""", unsafe_allow_html=True)