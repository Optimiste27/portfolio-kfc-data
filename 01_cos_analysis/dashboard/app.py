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
current_dir = os.path.dirname(os.path.abspath(__file__))  # dossier dashboard
root_dir = os.path.dirname(current_dir)  # dossier 01_cos_analysis
src_dir = os.path.join(root_dir, 'src')
data_dir = os.path.join(root_dir, 'data', 'raw')
reports_dir = os.path.join(root_dir, 'reports')

# Ajoute les chemins nécessaires
sys.path.insert(0, src_dir)
sys.path.insert(0, root_dir)

# Crée les dossiers nécessaires
os.makedirs(reports_dir, exist_ok=True)
# === FIN CONFIGURATION ===

# Configuration Streamlit
st.set_page_config(
    page_title="Dashboard COS KFC",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre
st.title("🏪 Dashboard d'Analyse COS - KFC")
st.markdown("### Surveillance des écarts de coût opérationnel")

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
    .critical-alert {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-alert {
        background-color: #d4edda;
        border: 1px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import du module PDF
try:
    from pdf_reporter import COSPDFReporter
    st.session_state.pdf_module_available = True
    st.session_state.pdf_reporter = COSPDFReporter(output_dir=reports_dir)
except ImportError as e:
    try:
        import reportlab
        st.session_state.pdf_module_available = False
    except ImportError:
        st.session_state.pdf_module_available = False

# Chargement intelligent des données
@st.cache_data
def load_and_prepare_data():
    """Charge et prépare les données avec détection automatique des colonnes"""
    try:
        # Cherche le fichier transactions
        trans_files = list(Path(data_dir).glob("*transaction*.csv"))
        if not trans_files:
            # Essayer avec un chemin différent
            trans_files = list(Path(root_dir).glob("**/*transaction*.csv"))
        
        if trans_files:
            latest_trans = max(trans_files, key=lambda x: x.stat().st_mtime)
            transactions = pd.read_csv(latest_trans)
            st.success(f"✅ Fichier chargé: {latest_trans.name}")
            
            # Afficher les colonnes pour debug
            st.sidebar.info(f"Colonnes: {', '.join(transactions.columns[:5])}...")
            
            # Cherche le fichier restaurants
            rest_files = list(Path(data_dir).glob("*restaurant*.csv"))
            if not rest_files:
                rest_files = list(Path(root_dir).glob("**/*restaurant*.csv"))
            
            if rest_files:
                latest_rest = max(rest_files, key=lambda x: x.stat().st_mtime)
                restaurants = pd.read_csv(latest_rest)
            else:
                # Crée un dataframe restaurants minimal
                restaurants = pd.DataFrame({
                    'restaurant_id': transactions['restaurant_id'].unique() if 'restaurant_id' in transactions.columns else range(1, 11),
                    'restaurant_name': transactions['restaurant_name'].unique() if 'restaurant_name' in transactions.columns else [f'Restaurant {i}' for i in range(1, 11)]
                })
            
            # ===== PRÉPARATION DES DONNÉES =====
            # 1. Standardiser les noms de colonnes
            columns_map = {}
            
            # Colonne date
            date_cols = ['date', 'Date', 'DATE', 'transaction_date']
            for col in date_cols:
                if col in transactions.columns:
                    columns_map[col] = 'date'
                    break
            
            # Colonne restaurant
            rest_cols = ['restaurant_name', 'restaurant', 'Restaurant', 'RESTAURANT_NAME']
            for col in rest_cols:
                if col in transactions.columns:
                    columns_map[col] = 'restaurant_name'
                    break
            
            # Colonne produit
            prod_cols = ['product_family', 'product', 'Product', 'PRODUCT_FAMILY']
            for col in prod_cols:
                if col in transactions.columns:
                    columns_map[col] = 'product_family'
                    break
            
            # Colonne écart
            gap_cols = ['gap_percentage', 'gap', 'GAP_PERCENTAGE', 'cos_gap_percentage']
            for col in gap_cols:
                if col in transactions.columns:
                    columns_map[col] = 'gap_percentage'
                    break
            
            # Colonne coût théorique
            th_cost_cols = ['theoretical_unit_cost', 'theoretical_cost', 'THEORETICAL_UNIT_COST']
            for col in th_cost_cols:
                if col in transactions.columns:
                    columns_map[col] = 'theoretical_unit_cost'
                    break
            
            # Colonne coût réel
            act_cost_cols = ['actual_unit_cost', 'actual_cost', 'ACTUAL_UNIT_COST']
            for col in act_cost_cols:
                if col in transactions.columns:
                    columns_map[col] = 'actual_unit_cost'
                    break
            
            # Colonne quantité vendue
            sold_cols = ['units_sold', 'quantity', 'UNITS_SOLD']
            for col in sold_cols:
                if col in transactions.columns:
                    columns_map[col] = 'units_sold'
                    break
            
            # Colonne gaspillage
            waste_cols = ['waste_kg', 'waste', 'WASTE_KG']
            for col in waste_cols:
                if col in transactions.columns:
                    columns_map[col] = 'waste_kg'
                    break
            
            # Colonne anomalies
            anomaly_cols = ['operational_anomaly', 'anomaly', 'OPERATIONAL_ANOMALY']
            for col in anomaly_cols:
                if col in transactions.columns:
                    columns_map[col] = 'operational_anomaly'
                    break
            
            # Colonne revenue
            revenue_cols = ['revenue', 'Revenue', 'REVENUE', 'total_revenue']
            for col in revenue_cols:
                if col in transactions.columns:
                    columns_map[col] = 'revenue'
                    break
            
            # Colonne score QSP
            qsp_cols = ['qsp_score', 'qsp', 'QSP_SCORE']
            for col in qsp_cols:
                if col in transactions.columns:
                    columns_map[col] = 'qsp_score'
                    break
            
            # 2. Renommer les colonnes
            transactions = transactions.rename(columns=columns_map)
            
            # 3. Conversion de date
            if 'date' in transactions.columns:
                transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce')
            else:
                # Créer une colonne date artificielle
                transactions['date'] = pd.date_range(
                    start='2024-01-01', 
                    periods=len(transactions), 
                    freq='D'
                )
            
            # 4. Calculer l'écart si nécessaire
            if 'gap_percentage' not in transactions.columns:
                if 'theoretical_unit_cost' in transactions.columns and 'actual_unit_cost' in transactions.columns:
                    # Calculer l'écart de coût
                    transactions['gap_percentage'] = ((transactions['actual_unit_cost'] - transactions['theoretical_unit_cost']) / 
                                                    transactions['theoretical_unit_cost']) * 100
                elif 'theoretical_cos' in transactions.columns and 'actual_cos' in transactions.columns:
                    transactions['gap_percentage'] = ((transactions['actual_cos'] - transactions['theoretical_cos']) / 
                                                    transactions['theoretical_cos']) * 100
                else:
                    # Écart aléatoire pour démo
                    transactions['gap_percentage'] = np.random.uniform(0, 10, len(transactions))
            
            # 5. Calculer la sévérité
            transactions['gap_severity'] = pd.cut(
                transactions['gap_percentage'], 
                bins=[-np.inf, 4, 6, np.inf], 
                labels=['Faible', 'Moyen', 'Élevé']
            )
            
            # 6. Calculer le revenue si nécessaire
            if 'revenue' not in transactions.columns:
                if 'selling_price' in transactions.columns and 'units_sold' in transactions.columns:
                    transactions['revenue'] = transactions['selling_price'] * transactions['units_sold']
                elif 'actual_unit_cost' in transactions.columns and 'units_sold' in transactions.columns:
                    # Estimation du revenue (coût * 2.5 pour marge)
                    transactions['revenue'] = transactions['actual_unit_cost'] * transactions['units_sold'] * 2.5
                else:
                    transactions['revenue'] = np.random.uniform(10, 100, len(transactions))
            
            # 7. Vérifier les anomalies
            if 'operational_anomaly' not in transactions.columns:
                transactions['operational_anomaly'] = np.random.choice(
                    [True, False], 
                    len(transactions), 
                    p=[0.15, 0.85]
                )
            
            # 8. Vérifier le gaspillage
            if 'waste_kg' not in transactions.columns:
                transactions['waste_kg'] = np.random.uniform(0, 2, len(transactions))
            
            return transactions, restaurants
            
        else:
            st.error("❌ Aucun fichier transactions trouvé")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        
        # Données d'exemple en cas d'erreur
        st.warning("⚠️ Utilisation de données d'exemple")
        n = 1000
        dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
        transactions = pd.DataFrame({
            'date': dates,
            'restaurant_id': np.random.choice(range(1, 11), n),
            'restaurant_name': np.random.choice([f'KFC Ville {i}' for i in range(1, 11)], n),
            'product_family': np.random.choice(['Poulet', 'Accompagnements', 'Boissons', 'Desserts'], n),
            'theoretical_unit_cost': np.random.uniform(1, 5, n),
            'actual_unit_cost': np.random.uniform(1.1, 6, n),
            'units_sold': np.random.randint(1, 50, n),
            'waste_kg': np.random.uniform(0, 1, n),
            'operational_anomaly': np.random.choice([True, False], n, p=[0.15, 0.85])
        })
        transactions['revenue'] = transactions['actual_unit_cost'] * transactions['units_sold'] * 2.5
        transactions['gap_percentage'] = ((transactions['actual_unit_cost'] - transactions['theoretical_unit_cost']) / 
                                        transactions['theoretical_unit_cost']) * 100
        transactions['gap_severity'] = pd.cut(
            transactions['gap_percentage'], 
            bins=[-np.inf, 4, 6, np.inf], 
            labels=['Faible', 'Moyen', 'Élevé']
        )
        
        restaurants = pd.DataFrame({
            'restaurant_id': range(1, 11),
            'restaurant_name': [f'KFC Ville {i}' for i in range(1, 11)],
            'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], 10)
        })
        
        return transactions, restaurants

# Chargement des données
with st.spinner("📥 Chargement et préparation des données..."):
    data = load_and_prepare_data()
    
    if data[0] is None:
        st.stop()
    
    transactions, restaurant_info = data
    
    # Afficher un aperçu des données
    with st.sidebar.expander("🔍 Aperçu données"):
        st.write(f"**Dimensions:** {transactions.shape[0]} lignes × {transactions.shape[1]} colonnes")
        st.write("**Colonnes principales:**")
        for col in ['date', 'restaurant_name', 'product_family', 'gap_percentage', 'revenue', 'operational_anomaly']:
            if col in transactions.columns:
                st.write(f"• {col}: ✅")
            else:
                st.write(f"• {col}: ❌")
        
        if 'gap_percentage' in transactions.columns:
            st.write(f"**Écart moyen:** {transactions['gap_percentage'].mean():.2f}%")

# Initialisation de l'état
if 'generated_reports' not in st.session_state:
    st.session_state.generated_reports = []

# Fonction de téléchargement PDF
def create_download_link(file_path, link_text="📥 Télécharger"):
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(file_path)}" style="background:#007bff;color:white;padding:6px 12px;text-decoration:none;border-radius:4px;font-weight:bold;">{link_text}</a>'
        return href
    except:
        return "❌ Fichier non trouvé"

# === SIDEBAR ===
with st.sidebar:
    st.header("📊 Navigation & Filtres")
    
    # Filtre restaurant
    if 'restaurant_name' in transactions.columns:
        restaurant_options = ["Tous"] + sorted(transactions['restaurant_name'].unique().tolist())
    else:
        restaurant_options = ["Tous"] + [f"Restaurant {i}" for i in range(1, 11)]
    
    selected_restaurant = st.selectbox("🏪 Restaurant", restaurant_options)
    
    # Filtre produit
    if 'product_family' in transactions.columns:
        product_options = ["Toutes"] + sorted(transactions['product_family'].unique().tolist())
    else:
        product_options = ["Toutes", "Poulet", "Accompagnements", "Boissons", "Desserts"]
    
    selected_product = st.selectbox("🍗 Produit", product_options)
    
    # Filtre date
    if 'date' in transactions.columns:
        min_date = transactions['date'].min().date()
        max_date = transactions['date'].max().date()
    else:
        min_date = datetime(2024, 1, 1).date()
        max_date = datetime(2024, 6, 28).date()
    
    date_range = st.date_input("📅 Période", [min_date, max_date])
    
    # Filtres avancés
    st.markdown("---")
    st.markdown("### 🎯 Filtres avancés")
    
    if 'gap_severity' in transactions.columns:
        severity_options = transactions['gap_severity'].cat.categories.tolist()
    else:
        severity_options = ['Faible', 'Moyen', 'Élevé']
    
    severity_filter = st.multiselect("Sévérité", severity_options, default=severity_options)
    
    if 'operational_anomaly' in transactions.columns:
        show_anomalies = st.checkbox("Anomalies uniquement", False)
    else:
        show_anomalies = False
    
    # Informations
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_transactions = len(transactions)
    unique_restaurants = transactions['restaurant_name'].nunique() if 'restaurant_name' in transactions.columns else 0
    unique_products = transactions['product_family'].nunique() if 'product_family' in transactions.columns else 0
    
    if 'gap_percentage' in transactions.columns:
        avg_gap = transactions['gap_percentage'].mean()
        gap_text = f"{avg_gap:.2f}%"
    else:
        gap_text = "N/A"
    
    st.info(f"""
    **Données:**
    • {total_transactions:,} transactions
    • {unique_restaurants} restaurants
    • {unique_products} produits
    • Écart moyen: {gap_text}
    """)

# === FILTRAGE DES DONNÉES ===
filtered_data = transactions.copy()

# Filtre restaurant
if selected_restaurant != "Tous":
    filtered_data = filtered_data[filtered_data['restaurant_name'] == selected_restaurant]

# Filtre produit
if selected_product != "Toutes":
    filtered_data = filtered_data[filtered_data['product_family'] == selected_product]

# Filtre date
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_data = filtered_data[
        (filtered_data['date'] >= pd.Timestamp(start_date)) & 
        (filtered_data['date'] <= pd.Timestamp(end_date))
    ]

# Filtre sévérité
if 'gap_severity' in filtered_data.columns:
    filtered_data = filtered_data[filtered_data['gap_severity'].isin(severity_filter)]

# Filtre anomalies
if show_anomalies and 'operational_anomaly' in filtered_data.columns:
    filtered_data = filtered_data[filtered_data['operational_anomaly'] == True]

# === MÉTRIQUES KPI ===
st.markdown("## 📈 Métriques Clés")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if 'gap_percentage' in filtered_data.columns:
        avg_gap = filtered_data['gap_percentage'].mean()
        status = "✅" if avg_gap <= 4 else "⚠️" if avg_gap <= 6 else "❌"
        st.metric(
            label=f"Écart COS moyen {status}",
            value=f"{avg_gap:.2f}%",
            delta=f"{(avg_gap - 4.0):+.2f}%" if avg_gap > 4 else None,
            delta_color="inverse"
        )
    else:
        st.metric("Écart COS moyen", "N/A")

with col2:
    if 'actual_unit_cost' in filtered_data.columns and 'units_sold' in filtered_data.columns:
        total_cost = (filtered_data['actual_unit_cost'] * filtered_data['units_sold']).sum()
        if 'theoretical_unit_cost' in filtered_data.columns:
            theoretical_cost = (filtered_data['theoretical_unit_cost'] * filtered_data['units_sold']).sum()
            cost_gap = total_cost - theoretical_cost
        else:
            cost_gap = 0
        st.metric("Surcoût total", f"{cost_gap:,.0f} €")
    else:
        st.metric("Surcoût total", "N/A")

with col3:
    if 'operational_anomaly' in filtered_data.columns:
        anomaly_rate = filtered_data['operational_anomaly'].mean() * 100
        st.metric("Taux d'anomalies", f"{anomaly_rate:.1f}%")
    else:
        st.metric("Taux d'anomalies", "N/A")

with col4:
    if 'waste_kg' in filtered_data.columns and 'theoretical_unit_cost' in filtered_data.columns:
        waste_cost = (filtered_data['waste_kg'] * filtered_data['theoretical_unit_cost']).sum()
        st.metric("Coût gaspillage", f"{waste_cost:,.0f} €")
    else:
        st.metric("Coût gaspillage", "N/A")

# === ONGLETS ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Vue d'ensemble", 
    "🏪 Par restaurant", 
    "🍗 Par produit", 
    "📊 Détails",
    "📄 Rapports PDF"
])

# Onglet 1: Vue d'ensemble
with tab1:
    st.subheader("📊 Vue d'ensemble")
    
    if 'gap_percentage' in filtered_data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.hist(filtered_data['gap_percentage'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(4.0, color='red', linestyle='--', label='Cible (4.0%)')
            ax1.set_xlabel('Écart COS (%)')
            ax1.set_ylabel('Fréquence')
            ax1.set_title('Distribution des écarts COS')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col2:
            # Top restaurants par écart
            if 'restaurant_name' in filtered_data.columns:
                restaurant_stats = filtered_data.groupby('restaurant_name')['gap_percentage'].mean().round(2)
                top5 = restaurant_stats.nlargest(5)
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                colors = ['red' if x > 6 else 'orange' for x in top5.values]
                ax2.barh(top5.index, top5.values, color=colors, edgecolor='black')
                ax2.set_xlabel('Écart moyen (%)')
                ax2.set_title('TOP 5 - Plus grands écarts')
                for i, (idx, val) in enumerate(top5.items()):
                    ax2.text(val + 0.1, i, f"{val}%", va='center')
                st.pyplot(fig2)
                plt.close(fig2)
    
    # Évolution temporelle
    st.subheader("📅 Évolution temporelle")
    
    if 'date' in filtered_data.columns and 'gap_percentage' in filtered_data.columns:
        daily_data = filtered_data.groupby('date')['gap_percentage'].mean().reset_index()
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['gap_percentage'],
            mode='lines+markers',
            name='Écart COS moyen',
            line=dict(color='red', width=2)
        ))
        
        fig3.update_layout(
            title='Évolution quotidienne',
            yaxis=dict(title='Écart COS (%)'),
            xaxis=dict(title='Date'),
            height=400
        )
        
        st.plotly_chart(fig3, width='stretch')

# Onglet 2: Par restaurant
with tab2:
    st.subheader("🏪 Analyse par restaurant")
    
    if 'restaurant_name' in filtered_data.columns:
        # Statistiques par restaurant
        agg_dict = {}
        if 'gap_percentage' in filtered_data.columns:
            agg_dict['gap_percentage'] = ['mean', 'std', 'count']
        if 'operational_anomaly' in filtered_data.columns:
            agg_dict['operational_anomaly'] = 'mean'
        if 'revenue' in filtered_data.columns:
            agg_dict['revenue'] = 'sum'
        
        if agg_dict:
            restaurant_stats = filtered_data.groupby('restaurant_name').agg(agg_dict).round(2)
            
            # Aplatir les colonnes multi-index
            restaurant_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in restaurant_stats.columns]
            
            # Renommer
            rename_map = {}
            if 'gap_percentage_mean' in restaurant_stats.columns:
                rename_map['gap_percentage_mean'] = 'Écart moyen %'
            if 'gap_percentage_std' in restaurant_stats.columns:
                rename_map['gap_percentage_std'] = 'Std écart'
            if 'gap_percentage_count' in restaurant_stats.columns:
                rename_map['gap_percentage_count'] = 'Nb transactions'
            if 'operational_anomaly_mean' in restaurant_stats.columns:
                rename_map['operational_anomaly_mean'] = 'Taux anomalies'
            if 'revenue_sum' in restaurant_stats.columns:
                rename_map['revenue_sum'] = 'CA total'
            
            restaurant_stats = restaurant_stats.rename(columns=rename_map)
            
            st.dataframe(restaurant_stats, width='stretch')
        else:
            st.info("Aucune statistique disponible pour les restaurants")

# Onglet 3: Par produit
with tab3:
    st.subheader("🍗 Analyse par produit")
    
    if 'product_family' in filtered_data.columns:
        # Statistiques par produit
        agg_dict = {}
        if 'gap_percentage' in filtered_data.columns:
            agg_dict['gap_percentage'] = ['mean', 'std']
        if 'revenue' in filtered_data.columns:
            agg_dict['revenue'] = 'sum'
        if 'units_sold' in filtered_data.columns:
            agg_dict['units_sold'] = 'sum'
        
        if agg_dict:
            product_stats = filtered_data.groupby('product_family').agg(agg_dict).round(2)
            
            # Aplatir et renommer
            product_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in product_stats.columns]
            
            rename_map = {}
            if 'gap_percentage_mean' in product_stats.columns:
                rename_map['gap_percentage_mean'] = 'Écart moyen %'
            if 'gap_percentage_std' in product_stats.columns:
                rename_map['gap_percentage_std'] = 'Std écart'
            if 'revenue_sum' in product_stats.columns:
                rename_map['revenue_sum'] = 'CA total'
            if 'units_sold_sum' in product_stats.columns:
                rename_map['units_sold_sum'] = 'Unités vendues'
            
            product_stats = product_stats.rename(columns=rename_map)
            
            st.dataframe(product_stats, width='stretch')

# Onglet 4: Détails
with tab4:
    st.subheader("📊 Détails des transactions")
    
    # Colonnes à afficher
    display_cols = []
    if 'date' in filtered_data.columns:
        display_cols.append('date')
    if 'restaurant_name' in filtered_data.columns:
        display_cols.append('restaurant_name')
    if 'product_family' in filtered_data.columns:
        display_cols.append('product_family')
    if 'gap_percentage' in filtered_data.columns:
        display_cols.append('gap_percentage')
    if 'gap_severity' in filtered_data.columns:
        display_cols.append('gap_severity')
    if 'revenue' in filtered_data.columns:
        display_cols.append('revenue')
    
    if display_cols:
        st.dataframe(filtered_data[display_cols], width='stretch', height=400)
        
        # Export
        csv = filtered_data[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter CSV",
            data=csv,
            file_name=f"cos_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Aucune donnée à afficher")

# Onglet 5: Rapports PDF
with tab5:
    st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
    st.markdown("## 📄 Générateur de rapports PDF")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.get('pdf_module_available', False):
        st.error("Module PDF non disponible. Installez reportlab: `pip install reportlab`")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Rapport hebdomadaire", use_container_width=True):
                with st.spinner("Génération..."):
                    try:
                        report_path = st.session_state.pdf_reporter.generate_weekly_report(
                            transactions,
                            restaurant_info,
                            week_number=1
                        )
                        if report_path:
                            st.success("✅ Rapport généré!")
                            st.markdown(create_download_link(report_path), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erreur: {str(e)[:100]}")
        
        with col2:
            if 'restaurant_name' in restaurant_info.columns:
                selected_rest = st.selectbox(
                    "Restaurant pour rapport",
                    restaurant_info['restaurant_name'].unique()
                )
                if st.button(f"📋 Rapport {selected_rest}", use_container_width=True):
                    with st.spinner("Génération..."):
                        try:
                            report_path = st.session_state.pdf_reporter.generate_restaurant_report(
                                transactions,
                                restaurant_info,
                                selected_rest
                            )
                            if report_path:
                                st.success("✅ Rapport généré!")
                                st.markdown(create_download_link(report_path), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erreur: {str(e)[:100]}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    🍗 Dashboard COS KFC • {datetime.now().strftime('%d/%m/%Y %H:%M')} • 
    {len(filtered_data):,} transactions • PDF: {'✅' if st.session_state.get('pdf_module_available', False) else '❌'}
</div>
""", unsafe_allow_html=True)