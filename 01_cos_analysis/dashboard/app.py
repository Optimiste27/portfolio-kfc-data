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

# === CORRECTION DES CHEMINS ===
# Ajoute le chemin vers le dossier parent et src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

# Ajoute les chemins nécessaires
sys.path.insert(0, src_dir)
sys.path.insert(0, parent_dir)
# === FIN CORRECTION ===

# Configuration
st.set_page_config(
    page_title="Dashboard COS KFC",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre
st.title("🏪 Dashboard d'Analyse COS - KFC")
st.markdown("### Surveillance des écarts de coût opérationnel")

# Style CSS personnalisé
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Import des modules de génération PDF
try:
    from pdf_reporter import COSPDFReporter
    st.session_state.pdf_module_available = True
except ImportError as e:
    # Vérifie si c'est reportlab qui manque
    try:
        import reportlab
        # Si reportlab est disponible, c'est un autre problème
        st.session_state.pdf_module_available = False
        st.error(f"❌ Erreur d'importation de pdf_reporter: {e}")
    except ImportError:
        # ReportLab n'est pas installé
        st.session_state.pdf_module_available = False
        st.warning("⚠️ Module PDF non disponible. Installez reportlab: `pip install reportlab`")

# Chargement des données
@st.cache_data
def load_data():
    """Charge les données avec recherche intelligente des fichiers"""
    # Cherche les fichiers les plus récents
    data_dir = Path("../data/raw")
    if not data_dir.exists():
        # Essaye avec le chemin courant
        data_dir = Path("data/raw")
    
    if data_dir.exists():
        csv_files = list(data_dir.glob("*transactions*.csv"))
        if csv_files:
            latest_transactions = max(csv_files, key=lambda x: x.stat().st_mtime)
            transactions = pd.read_csv(latest_transactions)
            
            # Cherche le fichier restaurants correspondant
            rest_files = list(data_dir.glob("*restaurants*.csv"))
            if rest_files:
                latest_restaurants = max(rest_files, key=lambda x: x.stat().st_mtime)
                restaurants = pd.read_csv(latest_restaurants)
            else:
                # Crée un dataframe restaurants minimal
                restaurants = pd.DataFrame({
                    'restaurant_id': transactions['restaurant_id'].unique(),
                    'restaurant_name': transactions['restaurant_name'].unique()
                })
            
            # Conversion de date
            transactions['date'] = pd.to_datetime(transactions['date'])
            
            return transactions, restaurants
    
    # Fallback vers les chemins originaux
    try:
        transactions = pd.read_csv('../data/raw/transactions_final_20260124_1112.csv')
        restaurants = pd.read_csv('../data/raw/restaurants_final_20260124_1112.csv')
        transactions['date'] = pd.to_datetime(transactions['date'])
        return transactions, restaurants
    except:
        # Si tout échoue, crée des données d'exemple
        st.warning("⚠️ Fichiers de données non trouvés. Utilisation de données d'exemple.")
        n_transactions = 1000
        dates = pd.date_range(start='2024-01-01', end='2024-06-28', periods=n_transactions)
        transactions = pd.DataFrame({
            'date': dates,
            'restaurant_id': np.random.choice(range(1, 11), n_transactions),
            'restaurant_name': np.random.choice([f'KFC Ville {i}' for i in range(1, 11)], n_transactions),
            'product_family': np.random.choice(['Poulet', 'Accompagnements', 'Boissons', 'Desserts'], n_transactions),
            'theoretical_unit_cost': np.random.uniform(1, 5, n_transactions),
            'actual_unit_cost': np.random.uniform(1, 6, n_transactions),
            'selling_price': np.random.uniform(3, 10, n_transactions),
            'units_sold': np.random.randint(1, 100, n_transactions),
            'revenue': lambda df: df['selling_price'] * df['units_sold'],
            'waste_kg': np.random.uniform(0, 2, n_transactions),
            'qsp_score': np.random.uniform(5, 10, n_transactions),
            'operational_anomaly': np.random.choice([True, False], n_transactions, p=[0.2, 0.8])
        })
        transactions['theoretical_cos'] = (transactions['theoretical_unit_cost'] * transactions['units_sold']) / transactions['revenue']
        transactions['actual_cos'] = (transactions['actual_unit_cost'] * transactions['units_sold']) / transactions['revenue']
        transactions['cos_gap'] = transactions['actual_cos'] - transactions['theoretical_cos']
        transactions['gap_percentage'] = (transactions['cos_gap'] / transactions['theoretical_cos']) * 100
        transactions['gap_severity'] = pd.cut(
            transactions['gap_percentage'], 
            bins=[-np.inf, 4, 6, np.inf], 
            labels=['Faible', 'Moyen', 'Élevé']
        )
        
        restaurants = pd.DataFrame({
            'restaurant_id': range(1, 11),
            'restaurant_name': [f'KFC Ville {i}' for i in range(1, 11)]
        })
        
        return transactions, restaurants

# Chargement des données
transactions, restaurant_info = load_data()

# Initialisation du générateur PDF dans session_state
if 'pdf_reporter' not in st.session_state and st.session_state.get('pdf_module_available', False):
    st.session_state.pdf_reporter = COSPDFReporter(output_dir="../reports")

# Initialisation historique des rapports
if 'generated_reports' not in st.session_state:
    st.session_state.generated_reports = []

# Fonction pour créer un lien de téléchargement PDF
def create_download_link(file_path, link_text="📥 Télécharger"):
    """Crée un lien de téléchargement pour un fichier PDF"""
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(file_path)}" style="background-color:#007bff;color:white;padding:8px 16px;text-decoration:none;border-radius:4px;font-weight:bold;display:inline-block;margin:5px 0;">{link_text}</a>'
        return href
    except Exception as e:
        return f"<span style='color:red;'>❌ Erreur: {str(e)[:50]}...</span>"

# Sidebar
with st.sidebar:
    st.header("📊 Navigation & Filtres")
    
    st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/b/bf/KFC_logo.svg/320px-KFC_logo.svg.png", 
             width=200, use_column_width=True)
    
    st.markdown("---")
    
    # Filtre restaurant
    selected_restaurant = st.selectbox(
        "🏪 Sélectionnez un restaurant",
        ["Tous"] + sorted(transactions['restaurant_name'].unique().tolist())
    )
    
    # Filtre produit
    selected_product = st.selectbox(
        "🍗 Sélectionnez une famille de produit",
        ["Toutes"] + sorted(transactions['product_family'].unique().tolist())
    )
    
    # Filtre date
    date_range = st.date_input(
        "📅 Période",
        [transactions['date'].min().date(), transactions['date'].max().date()]
    )
    
    # Filtre sévérité
    st.markdown("---")
    st.markdown("### 🎯 Filtres avancés")
    
    severity_filter = st.multiselect(
        "Niveau de sévérité",
        ['Faible', 'Moyen', 'Élevé'],
        default=['Faible', 'Moyen', 'Élevé']
    )
    
    show_only_anomalies = st.checkbox("Afficher uniquement les anomalies", False)
    
    # Info données
    st.markdown("---")
    st.markdown("### 📊 Informations")
    st.info(f"""
    **Données chargées:**
    - {len(transactions):,} transactions
    - {transactions['restaurant_name'].nunique()} restaurants
    - Du {transactions['date'].min().strftime('%d/%m/%Y')}
    - Au {transactions['date'].max().strftime('%d/%m/%Y')}
    """)
    
    # Bouton rechargement
    if st.button("🔄 Actualiser les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Filtrage des données
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
if severity_filter:
    filtered_data = filtered_data[filtered_data['gap_severity'].isin(severity_filter)]

# Filtre anomalies
if show_only_anomalies:
    filtered_data = filtered_data[filtered_data['operational_anomaly'] == True]

# Métriques KPI
st.markdown("## 📈 Métriques Clés")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_gap = filtered_data['gap_percentage'].mean()
    st.metric(
        label="Écart COS moyen",
        value=f"{avg_gap:.2f}%",
        delta=f"{(avg_gap - 4.0):.2f}%" if selected_restaurant == "Tous" else None,
        delta_color="inverse"
    )

with col2:
    total_cost = (filtered_data['actual_unit_cost'] * filtered_data['units_sold']).sum()
    theoretical_cost = (filtered_data['theoretical_unit_cost'] * filtered_data['units_sold']).sum()
    cost_gap = total_cost - theoretical_cost
    st.metric(
        label="Surcoût total",
        value=f"{cost_gap:,.0f} €",
        delta_color="inverse"
    )

with col3:
    anomaly_rate = filtered_data['operational_anomaly'].mean() * 100
    st.metric(
        label="Taux d'anomalies",
        value=f"{anomaly_rate:.1f}%",
        delta_color="inverse"
    )

with col4:
    waste_cost = (filtered_data['waste_kg'] * filtered_data['theoretical_unit_cost']).sum()
    st.metric(
        label="Coût du gaspillage",
        value=f"{waste_cost:,.0f} €"
    )

# Onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Vue d'ensemble", 
    "🏪 Par restaurant", 
    "🍗 Par produit", 
    "📊 Détails",
    "📄 Rapports PDF"
])

# Onglet 1: Vue d'ensemble
with tab1:
    st.subheader("📊 Vue d'ensemble des performances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des écarts
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
        # Top 5 restaurants
        restaurant_stats = filtered_data.groupby('restaurant_name').agg({
            'gap_percentage': 'mean',
            'operational_anomaly': 'mean'
        }).round(2)
        
        top5 = restaurant_stats.nlargest(5, 'gap_percentage')
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors = ['red' if x > 6 else 'orange' for x in top5['gap_percentage']]
        ax2.barh(top5.index, top5['gap_percentage'], color=colors, edgecolor='black')
        ax2.set_xlabel('Écart COS moyen (%)')
        ax2.set_title('TOP 5 - Restaurants avec plus d\'écarts')
        for i, (idx, val) in enumerate(top5['gap_percentage'].items()):
            ax2.text(val + 0.1, i, f"{val}%", va='center')
        st.pyplot(fig2)
    
    # Évolution temporelle
    st.subheader("📅 Évolution temporelle")
    
    daily_data = filtered_data.groupby('date').agg({
        'gap_percentage': 'mean',
        'operational_anomaly': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['gap_percentage'],
        mode='lines+markers',
        name='Écart COS moyen',
        line=dict(color='red', width=2)
    ))
    
    fig3.add_trace(go.Bar(
        x=daily_data['date'],
        y=daily_data['operational_anomaly'] * 100,
        name='Taux anomalies (%)',
        yaxis='y2',
        marker_color='rgba(128, 0, 128, 0.3)'
    ))
    
    fig3.update_layout(
        title='Évolution quotidienne',
        yaxis=dict(title='Écart COS (%)'),
        yaxis2=dict(title='Taux anomalies (%)', overlaying='y', side='right'),
        xaxis=dict(title='Date'),
        height=400
    )
    
    st.plotly_chart(fig3, width='stretch')

# Onglet 2: Par restaurant
with tab2:
    st.subheader("🏪 Analyse détaillée par restaurant")
    
    # Performance détaillée par restaurant
    restaurant_perf = filtered_data.groupby('restaurant_name').agg({
        'gap_percentage': ['mean', 'std', 'count'],
        'operational_anomaly': 'mean',
        'waste_kg': 'mean',
        'revenue': 'sum'
    }).round(2)
    
    restaurant_perf.columns = ['Écart moyen %', 'Std écart', 'Nb transactions', 
                               'Taux anomalies', 'Gaspillage moyen', 'CA total']
    
    # Calcul de la classification
    restaurant_perf['Classification'] = pd.cut(
        restaurant_perf['Écart moyen %'],
        bins=[-np.inf, 4, 6, np.inf],
        labels=['✅ Bon', '⚠️ À surveiller', '❌ Critique']
    )
    
    st.dataframe(
        restaurant_perf.sort_values('Écart moyen %', ascending=False),
        width='stretch'
    )
    
    # Graphique comparatif
    fig = go.Figure(data=[
        go.Bar(
            name='Écart moyen',
            x=restaurant_perf.index,
            y=restaurant_perf['Écart moyen %'],
            marker_color=['red' if x > 6 else 'orange' if x > 4 else 'green' for x in restaurant_perf['Écart moyen %']]
        ),
        go.Scatter(
            name='Taux anomalies',
            x=restaurant_perf.index,
            y=restaurant_perf['Taux anomalies'] * 100,
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='purple', width=2)
        )
    ])
    
    fig.update_layout(
        title='Performance par restaurant',
        yaxis=dict(title='Écart COS moyen (%)'),
        yaxis2=dict(title='Taux anomalies (%)', overlaying='y', side='right'),
        xaxis=dict(tickangle=45),
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Détails pour un restaurant spécifique
    st.subheader("🔍 Zoom sur un restaurant")
    
    selected_rest_detail = st.selectbox(
        "Choisir un restaurant pour plus de détails",
        restaurant_perf.index.tolist(),
        key="rest_detail"
    )
    
    if selected_rest_detail:
        rest_data = filtered_data[filtered_data['restaurant_name'] == selected_rest_detail]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des écarts
            fig_rest, ax_rest = plt.subplots(figsize=(8, 5))
            ax_rest.hist(rest_data['gap_percentage'], bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
            ax_rest.axvline(rest_data['gap_percentage'].mean(), color='red', linestyle='--', 
                           label=f'Moyenne: {rest_data["gap_percentage"].mean():.2f}%')
            ax_rest.set_xlabel('Écart COS (%)')
            ax_rest.set_ylabel('Fréquence')
            ax_rest.set_title(f'Distribution - {selected_rest_detail}')
            ax_rest.legend()
            ax_rest.grid(True, alpha=0.3)
            st.pyplot(fig_rest)
        
        with col2:
            # Performance par produit
            product_stats = rest_data.groupby('product_family').agg({
                'gap_percentage': 'mean',
                'revenue': 'sum'
            }).round(2)
            
            st.dataframe(
                product_stats.sort_values('gap_percentage', ascending=False),
                width='stretch'
            )

# Onglet 3: Par produit
with tab3:
    st.subheader("🍗 Analyse par famille de produit")
    
    # Performance par produit
    product_perf = filtered_data.groupby('product_family').agg({
        'gap_percentage': ['mean', 'std'],
        'waste_kg': 'mean',
        'revenue': 'sum',
        'units_sold': 'sum'
    }).round(2)
    
    product_perf.columns = ['Écart moyen %', 'Std écart', 'Gaspillage moyen', 'CA total', 'Unités vendues']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            product_perf.sort_values('Écart moyen %', ascending=False),
            width='stretch'
        )
    
    with col2:
        # Graphique produits
        fig, ax = plt.subplots(figsize=(10, 6))
        product_perf_sorted = product_perf.sort_values('Écart moyen %', ascending=True)
        bars = ax.barh(product_perf_sorted.index, product_perf_sorted['Écart moyen %'],
                      color='lightcoral', edgecolor='black')
        ax.set_xlabel('Écart COS moyen (%)')
        ax.set_title('Performance par famille de produit')
        ax.axvline(x=4.0, color='red', linestyle='--', alpha=0.5, label='Moyenne')
        ax.legend()
        st.pyplot(fig)
    
    # Heatmap produit x restaurant
    st.subheader("🎯 Matrice Produit x Restaurant")
    
    pivot_data = filtered_data.pivot_table(
        values='gap_percentage',
        index='product_family',
        columns='restaurant_name',
        aggfunc='mean'
    ).round(2)
    
    fig_heatmap = px.imshow(
        pivot_data,
        labels=dict(x="Restaurant", y="Produit", color="Écart COS"),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='RdYlGn_r',
        title='Écart COS moyen par produit et restaurant'
    )
    
    st.plotly_chart(fig_heatmap, width='stretch')

# Onglet 4: Détails
with tab4:
    st.subheader("📊 Détails des transactions")
    
    # Filtres avancés
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_gap = st.slider("Écart minimum (%)", 0.0, 20.0, 0.0, key="min_gap")
    with col2:
        max_gap = st.slider("Écart maximum (%)", 0.0, 20.0, 20.0, key="max_gap")
    with col3:
        sort_by = st.selectbox(
            "Trier par",
            ['Écart décroissant', 'Écart croissant', 'Date récente', 'Date ancienne']
        )
    
    # Application des filtres
    detailed_data = filtered_data.copy()
    detailed_data = detailed_data[
        (detailed_data['gap_percentage'] >= min_gap) & 
        (detailed_data['gap_percentage'] <= max_gap)
    ]
    
    # Tri
    if sort_by == 'Écart décroissant':
        detailed_data = detailed_data.sort_values('gap_percentage', ascending=False)
    elif sort_by == 'Écart croissant':
        detailed_data = detailed_data.sort_values('gap_percentage', ascending=True)
    elif sort_by == 'Date récente':
        detailed_data = detailed_data.sort_values('date', ascending=False)
    else:
        detailed_data = detailed_data.sort_values('date', ascending=True)
    
    # Affichage des données
    st.dataframe(
        detailed_data[[
            'date', 'restaurant_name', 'product_family', 
            'gap_percentage', 'gap_severity', 'operational_anomaly',
            'waste_kg', 'qsp_score', 'revenue'
        ]],
        width='stretch',
        height=400
    )
    
    # Bouton d'export
    csv = detailed_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exporter les données filtrées",
        data=csv,
        file_name=f"cos_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Statistiques descriptives
    with st.expander("📈 Statistiques descriptives"):
        st.write("**Résumé des écarts COS:**")
        st.write(detailed_data['gap_percentage'].describe())
        
        st.write("**Répartition des sévérités:**")
        severity_counts = detailed_data['gap_severity'].value_counts()
        st.bar_chart(severity_counts)

# Onglet 5: Rapports PDF
with tab5:
    st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
    st.markdown("## 📄 Générateur de rapports PDF")
    st.markdown("Générez des rapports professionnels en PDF pour analyse et partage")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.get('pdf_module_available', False):
        st.error("""
        **Module PDF non disponible**
        
        Pour activer la génération de PDF, installez le module reportlab :
        ```bash
        pip install reportlab
        ```
        Redémarrez le dashboard après l'installation.
        """)
    else:
        # Vérification du dossier reports
        reports_dir = Path("../reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Section de génération de rapports
        st.markdown("### 🚀 Générer des rapports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📅 Rapports temporels")
            
            if st.button("📊 Rapport hebdomadaire", type="primary", use_container_width=True):
                with st.spinner("Génération du rapport hebdomadaire..."):
                    try:
                        report_path = st.session_state.pdf_reporter.generate_weekly_report(
                            transactions,
                            restaurant_info,
                            week_number=1
                        )
                        
                        if report_path and os.path.exists(report_path):
                            st.success(f"✅ Rapport généré: {os.path.basename(report_path)}")
                            st.markdown(create_download_link(report_path, "📥 Télécharger le rapport"), unsafe_allow_html=True)
                            
                            # Ajout à l'historique
                            st.session_state.generated_reports.append({
                                'type': 'weekly',
                                'name': 'Rapport hebdomadaire (Semaine 1)',
                                'path': report_path,
                                'timestamp': datetime.now()
                            })
                        else:
                            st.error("❌ Échec de la génération du rapport")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)[:100]}")
            
            if st.button("📈 Rapport mensuel", use_container_width=True):
                with st.spinner("Génération du rapport mensuel..."):
                    try:
                        current_month = datetime.now().month
                        current_year = datetime.now().year
                        
                        report_path = st.session_state.pdf_reporter.generate_monthly_report(
                            transactions,
                            restaurant_info,
                            month=current_month,
                            year=current_year
                        )
                        
                        if report_path and os.path.exists(report_path):
                            st.success(f"✅ Rapport mensuel généré!")
                            st.markdown(create_download_link(report_path, "📥 Télécharger le rapport"), unsafe_allow_html=True)
                            
                            # Ajout à l'historique
                            st.session_state.generated_reports.append({
                                'type': 'monthly',
                                'name': f'Rapport mensuel {current_month}/{current_year}',
                                'path': report_path,
                                'timestamp': datetime.now()
                            })
                        else:
                            st.error("❌ Échec de la génération du rapport")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)[:100]}")
        
        with col2:
            st.markdown("#### 🏪 Rapports par restaurant")
            
            # Sélection du restaurant
            restaurant_options = restaurant_info['restaurant_name'].unique()
            selected_for_report = st.selectbox(
                "Choisir un restaurant",
                restaurant_options,
                key="pdf_restaurant_select"
            )
            
            if st.button(f"📋 Générer rapport pour {selected_for_report}", type="primary", use_container_width=True):
                with st.spinner(f"Génération du rapport pour {selected_for_report}..."):
                    try:
                        report_path = st.session_state.pdf_reporter.generate_restaurant_report(
                            transactions,
                            restaurant_info,
                            selected_for_report
                        )
                        
                        if report_path and os.path.exists(report_path):
                            st.success(f"✅ Rapport pour {selected_for_report} généré!")
                            st.markdown(create_download_link(report_path, f"📥 Télécharger {selected_for_report}"), unsafe_allow_html=True)
                            
                            # Ajout à l'historique
                            st.session_state.generated_reports.append({
                                'type': 'restaurant',
                                'name': selected_for_report,
                                'path': report_path,
                                'timestamp': datetime.now()
                            })
                        else:
                            st.error("❌ Échec de la génération du rapport")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)[:100]}")
            
            # Bouton pour les 3 pires restaurants
            if st.button("⚠️ Rapports des 3 pires restaurants", use_container_width=True):
                with st.spinner("Identification et génération des rapports..."):
                    try:
                        # Identifie les 3 pires restaurants
                        restaurant_stats = transactions.groupby('restaurant_name').agg({
                            'gap_percentage': 'mean'
                        }).round(2)
                        
                        worst_restaurants = restaurant_stats.nlargest(3, 'gap_percentage').index.tolist()
                        
                        generated = []
                        for rest_name in worst_restaurants:
                            report_path = st.session_state.pdf_reporter.generate_restaurant_report(
                                transactions,
                                restaurant_info,
                                rest_name
                            )
                            if report_path and os.path.exists(report_path):
                                generated.append((rest_name, report_path))
                        
                        if generated:
                            st.success(f"✅ {len(generated)} rapports générés!")
                            for rest_name, report_path in generated:
                                with st.expander(f"📄 {rest_name}"):
                                    st.markdown(create_download_link(report_path), unsafe_allow_html=True)
                        else:
                            st.error("❌ Aucun rapport généré")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)[:100]}")
        
        with col3:
            st.markdown("#### 📊 Rapports personnalisés")
            
            # Rapport personnalisé basé sur les filtres actuels
            custom_name = st.text_input("Nom du rapport", 
                value=f"Analyse {selected_restaurant if selected_restaurant != 'Tous' else 'globale'}")
            
            if st.button("✨ Générer rapport personnalisé", type="primary", use_container_width=True):
                with st.spinner("Génération du rapport personnalisé..."):
                    try:
                        # Générer un rapport avec les données filtrées
                        if selected_restaurant != "Tous":
                            # Rapport spécifique au restaurant
                            report_path = st.session_state.pdf_reporter.generate_restaurant_report(
                                filtered_data,
                                restaurant_info,
                                selected_restaurant
                            )
                            report_type = f"Restaurant: {selected_restaurant}"
                        else:
                            # Rapport global ou générique
                            report_path = st.session_state.pdf_reporter.generate_weekly_report(
                                filtered_data,
                                restaurant_info,
                                week_number=1
                            )
                            report_type = "Analyse globale"
                        
                        if report_path and os.path.exists(report_path):
                            st.success(f"✅ Rapport '{custom_name}' généré!")
                            st.markdown(create_download_link(report_path, "📥 Télécharger le rapport"), unsafe_allow_html=True)
                            
                            # Ajout à l'historique
                            st.session_state.generated_reports.append({
                                'type': 'custom',
                                'name': custom_name,
                                'path': report_path,
                                'timestamp': datetime.now(),
                                'description': report_type
                            })
                        else:
                            st.error("❌ Échec de la génération du rapport")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)[:100]}")
            
            # Bouton pour tout générer
            if st.button("🚀 Générer package complet", type="secondary", use_container_width=True):
                with st.spinner("Génération du package complet..."):
                    try:
                        generated_reports = []
                        
                        # Rapport hebdomadaire
                        weekly_path = st.session_state.pdf_reporter.generate_weekly_report(
                            transactions,
                            restaurant_info,
                            week_number=1
                        )
                        if weekly_path:
                            generated_reports.append(("Hebdomadaire", weekly_path))
                        
                        # Rapport mensuel
                        monthly_path = st.session_state.pdf_reporter.generate_monthly_report(
                            transactions,
                            restaurant_info,
                            month=datetime.now().month,
                            year=datetime.now().year
                        )
                        if monthly_path:
                            generated_reports.append(("Mensuel", monthly_path))
                        
                        # Rapports pour les 3 pires restaurants
                        restaurant_stats = transactions.groupby('restaurant_name').agg({
                            'gap_percentage': 'mean'
                        }).round(2)
                        worst_restaurants = restaurant_stats.nlargest(3, 'gap_percentage').index.tolist()
                        
                        for rest_name in worst_restaurants:
                            rest_path = st.session_state.pdf_reporter.generate_restaurant_report(
                                transactions,
                                restaurant_info,
                                rest_name
                            )
                            if rest_path:
                                generated_reports.append((f"Restaurant: {rest_name}", rest_path))
                        
                        # Affichage des résultats
                        if generated_reports:
                            st.success(f"✅ {len(generated_reports)} rapports générés!")
                            
                            for report_name, report_path in generated_reports:
                                with st.expander(f"📄 {report_name}"):
                                    st.markdown(create_download_link(report_path), unsafe_allow_html=True)
                                    st.caption(f"Chemin: {report_path}")
                        else:
                            st.warning("⚠️ Aucun rapport généré")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)[:100]}")
        
        # Section Historique des rapports
        st.markdown("---")
        st.markdown("### 📚 Historique des rapports générés")
        
        if st.session_state.generated_reports:
            # Trier par date décroissante
            sorted_reports = sorted(
                st.session_state.generated_reports,
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
            # Afficher les 5 derniers rapports
            for i, report in enumerate(sorted_reports[:5]):
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.write(f"**{i+1}. {report['name']}**")
                        st.caption(f"Type: {report.get('type', 'N/A')} • {report.get('description', '')}")
                    
                    with col2:
                        st.caption(f"📅 {report['timestamp'].strftime('%d/%m/%Y %H:%M')}")
                        if 'path' in report and os.path.exists(report['path']):
                            file_size = os.path.getsize(report['path'])
                            st.caption(f"📏 {file_size:,} octets")
                    
                    with col3:
                        if 'path' in report and os.path.exists(report['path']):
                            st.markdown(create_download_link(report['path'], "📥"), unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Fichier manquant")
            
            # Bouton pour effacer l'historique
            if st.button("🗑️ Effacer l'historique", type="secondary"):
                st.session_state.generated_reports = []
                st.rerun()
        else:
            st.info("ℹ️ Aucun rapport généré pour le moment")
        
        # Section Aide
        with st.expander("❓ Guide d'utilisation", expanded=False):
            st.markdown("""
            ### 📖 Comment utiliser les rapports PDF
            
            **📅 Rapports temporels**
            - **Hebdomadaire**: Analyse détaillée de la semaine 1
            - **Mensuel**: Vue complète du mois en cours avec tendances
            
            **🏪 Rapports par restaurant**
            - Sélectionnez un restaurant spécifique
            - Rapport détaillé avec métriques et recommandations
            - **⚠️ Rapports des 3 pires restaurants**: Génère automatiquement pour les restaurants critiques
            
            **📊 Rapports personnalisés**
            - Basé sur vos filtres actuels (restaurant, produit, période)
            - Nom personnalisable
            - Contenu adapté aux données filtrées
            
            **🚀 Package complet**
            - Génère un ensemble complet de rapports (hebdo + mensuel + 3 pires restaurants)
            - Idéal pour les réunions de direction
            
            **💾 Téléchargement**
            - Cliquez sur "📥 Télécharger"
            - Les fichiers sont sauvegardés dans `reports/`
            - Historique des 5 derniers rapports conservés
            
            **🎯 Conseils d'utilisation**
            1. Commencez par générer les rapports des restaurants critiques
            2. Utilisez les rapports mensuels pour les réunions stratégiques
            3. Les rapports hebdomadaires sont parfaits pour le suivi opérationnel
            4. Personnalisez les rapports avec vos filtres avant génération
            """)

# Section recommandations
st.markdown("---")
st.subheader("🎯 Recommandations opérationnelles")

# Calcul des recommandations
if not filtered_data.empty:
    restaurant_perf_rec = filtered_data.groupby('restaurant_name').agg({
        'gap_percentage': 'mean',
        'revenue': 'sum'
    }).round(2)
    
    restaurant_perf_rec.columns = ['Écart moyen %', 'CA total']
    
    if selected_restaurant == "Tous" and not restaurant_perf_rec.empty and len(restaurant_perf_rec) > 1:
        # Recommandations globales
        worst_restaurant = restaurant_perf_rec.index[0]
        worst_gap = restaurant_perf_rec.iloc[0]['Écart moyen %']
        best_restaurant = restaurant_perf_rec.index[-1]
        best_gap = restaurant_perf_rec.iloc[-1]['Écart moyen %']
        
        # Calcul économie potentielle
        if worst_gap > 4 and restaurant_perf_rec.loc[worst_restaurant, 'CA total'] > 0:
            econ_potentielle = ((worst_gap - 4.0) * restaurant_perf_rec.loc[worst_restaurant, 'CA total'] / 100)
            econ_text = f"{econ_potentielle:,.0f} €"
        else:
            econ_text = "N/A"
        
        st.info(f"""
        **🎯 Priorité 1 : Audit sur {worst_restaurant}**  
        Écart de {worst_gap}% (cible: 4.0%) - Potentiel d'économie: {econ_text}
        
        **📈 Priorité 2 : Benchmark de {best_restaurant}**  
        Meilleure performance à {best_gap}% - Étudier et répliquer les bonnes pratiques
        
        **🍟 Priorité 3 : Focus produit critique**  
        Identifier la famille de produit avec le plus grand écart
        """)
        
        # Bouton rapide pour générer rapport
        if st.session_state.get('pdf_module_available', False):
            if st.button("📋 Générer rapport d'audit rapide", type="secondary"):
                with st.spinner("Génération du rapport..."):
                    try:
                        report_path = st.session_state.pdf_reporter.generate_restaurant_report(
                            transactions,
                            restaurant_info,
                            worst_restaurant
                        )
                        if report_path:
                            st.success("✅ Rapport d'audit généré!")
                            st.markdown(create_download_link(report_path, "📥 Télécharger le rapport"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erreur: {str(e)[:100]}")
                        
    elif selected_restaurant != "Tous":
        # Recommandations spécifiques au restaurant
        current_gap = avg_gap
        if current_gap > 6:
            st.error(f"🚨 **ACTION REQUISE** : Écart critique de {current_gap:.1f}%")
            st.write("**Actions immédiates recommandées:**")
            st.write("1. 🔍 Audit complet des processus opérationnels")
            st.write("2. 🎓 Formation équipe sur recettes standards")
            st.write("3. ⚙️ Vérification des équipements et calibrage")
            st.write("4. 📊 Analyse détaillée par produit critique")
            
        elif current_gap > 4:
            st.warning(f"⚠️  **AMÉLIORATION NÉCESSAIRE** : Écart de {current_gap:.1f}%")
            st.write("**Actions recommandées:**")
            st.write("1. 📈 Analyse détaillée par produit")
            st.write("2. 🎯 Renforcement des contrôles qualité")
            st.write("3. 🔄 Revue des procédures de préparation")
            st.write("4. 📋 Suivi rapproché des indicateurs")
        else:
            st.success(f"✅ **PERFORMANCE SATISFAISANTE** : Écart de {current_gap:.1f}%")
            st.write("**Actions de maintien:**")
            st.write("1. 🏆 Maintenir et partager les bonnes pratiques")
            st.write("2. 📊 Surveiller les tendances")
            st.write("3. 🔄 Continuer les formations régulières")
            st.write("4. 🎯 Définir des objectifs d'amélioration continue")

# Footer
st.markdown("---")
current_date = datetime.now().strftime('%d/%m/%Y %H:%M')
data_count = len(filtered_data)
pdf_status = '✅' if st.session_state.get('pdf_module_available', False) else '❌'

st.markdown(f"""
<div style='text-align: center; color: gray; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
    <div style='font-size: 0.9rem; margin-bottom: 10px;'>
        🍗 <b>Dashboard COS KFC</b> • Version 2.1 • {current_date}
    </div>
    <div style='font-size: 0.8rem; color: #666;'>
        📊 {data_count:,} transactions filtrées • 📄 Module PDF: {pdf_status} • 
        🏪 {filtered_data['restaurant_name'].nunique()} restaurants • 
        🍗 {filtered_data['product_family'].nunique()} produits
    </div>
    <div style='font-size: 0.7rem; margin-top: 10px; color: #888;'>
        Dashboard d'analyse des coûts opérationnels • Données à usage pédagogique
    </div>
    <div style='font-size: 0.7rem; margin-top: 5px; color: #aaa;'>
        Écart COS moyen: {avg_gap:.2f}% • Période: {filtered_data['date'].min().strftime('%d/%m/%y')} - {filtered_data['date'].max().strftime('%d/%m/%y')}
    </div>
</div>
""", unsafe_allow_html=True)