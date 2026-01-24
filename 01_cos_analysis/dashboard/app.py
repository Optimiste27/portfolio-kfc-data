import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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

# Chargement des données
@st.cache_data
def load_data():
    transactions = pd.read_csv('../data/raw/transactions_final_20260124_1112.csv')
    restaurants = pd.read_csv('../data/raw/restaurants_final_20260124_1112.csv')
    return transactions, restaurants

transactions, restaurant_info = load_data()

# Sidebar
st.sidebar.header("📊 Filtres")
selected_restaurant = st.sidebar.selectbox(
    "Sélectionnez un restaurant",
    ["Tous"] + sorted(transactions['restaurant_name'].unique().tolist())
)

selected_product = st.sidebar.selectbox(
    "Sélectionnez une famille de produit",
    ["Toutes"] + sorted(transactions['product_family'].unique().tolist())
)

date_range = st.sidebar.date_input(
    "Période",
    [datetime(2024, 1, 1), datetime(2024, 6, 28)]
)

# Filtrage
filtered_data = transactions.copy()
if selected_restaurant != "Tous":
    filtered_data = filtered_data[filtered_data['restaurant_name'] == selected_restaurant]
if selected_product != "Toutes":
    filtered_data = filtered_data[filtered_data['product_family'] == selected_product]

# Métriques KPI
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
tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue d'ensemble", "🏪 Par restaurant", "🍗 Par produit", "📊 Détails"])

with tab1:
    st.subheader("Vue d'ensemble des performances")
    
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

with tab2:
    st.subheader("Analyse par restaurant")
    
    # Performance détaillée par restaurant
    restaurant_perf = filtered_data.groupby('restaurant_name').agg({
        'gap_percentage': ['mean', 'std', 'count'],
        'operational_anomaly': 'mean',
        'waste_kg': 'mean',
        'revenue': 'sum'
    }).round(2)
    
    restaurant_perf.columns = ['Écart moyen %', 'Std écart', 'Nb transactions', 
                               'Taux anomalies', 'Gaspillage moyen', 'CA total']
    
    st.dataframe(
        restaurant_perf.sort_values('Écart moyen %', ascending=False),
        use_container_width=True
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
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Analyse par produit")
    
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
            use_container_width=True
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

with tab4:
    st.subheader("Détails des transactions")
    
    # Filtres avancés
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_gap = st.slider("Écart minimum (%)", 0.0, 20.0, 0.0)
    with col2:
        max_gap = st.slider("Écart maximum (%)", 0.0, 20.0, 20.0)
    with col3:
        show_anomalies = st.checkbox("Afficher seulement les anomalies", value=False)
    
    # Application des filtres
    detailed_data = filtered_data.copy()
    detailed_data = detailed_data[
        (detailed_data['gap_percentage'] >= min_gap) & 
        (detailed_data['gap_percentage'] <= max_gap)
    ]
    
    if show_anomalies:
        detailed_data = detailed_data[detailed_data['operational_anomaly'] == True]
    
    # Affichage des données
    st.dataframe(
        detailed_data[[
            'date', 'restaurant_name', 'product_family', 
            'gap_percentage', 'gap_severity', 'operational_anomaly',
            'waste_kg', 'qsp_score'
        ]].sort_values('gap_percentage', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Bouton d'export
    csv = detailed_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exporter les données filtrées",
        data=csv,
        file_name=f"cos_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Section recommandations
st.markdown("---")
st.subheader("🎯 Recommandations opérationnelles")

# Calcul des recommandations
if selected_restaurant == "Tous":
    # Recommandations globales
    worst_restaurant = restaurant_perf.index[0]
    worst_gap = restaurant_perf.iloc[0]['Écart moyen %']
    best_restaurant = restaurant_perf.index[-1]
    best_gap = restaurant_perf.iloc[-1]['Écart moyen %']
    
    st.info(f"""
    **Priorité 1 : Audit sur {worst_restaurant}**  
    Écart de {worst_gap}% (2x la moyenne) - Potentiel d'économie: {((worst_gap - 4.0) * restaurant_perf.loc[worst_restaurant, 'CA total'] / 10000):.0f} k€
    
    **Priorité 2 : Benchmark de {best_restaurant}**  
    Meilleure performance à {best_gap}% - Étudier et répliquer les bonnes pratiques
    
    **Priorité 3 : Focus sur Accompagnements**  
    Produit avec plus d'écart (4.5%) - Optimiser la gestion des frites
    """)
else:
    # Recommandations spécifiques au restaurant
    current_gap = avg_gap
    if current_gap > 6:
        st.error(f"🚨 **ACTION REQUISE** : Écart critique de {current_gap:.1f}%")
        st.write("Actions immédiates recommandées:")
        st.write("1. Audit complet des processus")
        st.write("2. Formation équipe sur recettes standards")
        st.write("3. Vérification équipements")
    elif current_gap > 4:
        st.warning(f"⚠️  **AMÉLIORATION NÉCESSAIRE** : Écart de {current_gap:.1f}%")
        st.write("Actions recommandées:")
        st.write("1. Analyse détaillée par produit")
        st.write("2. Renforcement contrôle qualité")
        st.write("3. Revue des procédures")
    else:
        st.success(f"✅ **PERFORMANCE SATISFAISANTE** : Écart de {current_gap:.1f}%")
        st.write("Maintenir et partager les bonnes pratiques")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Dashboard COS KFC • Données synthétiques • Généré le 24/01/2024</small><br>
    <small>⚠️ Données à usage pédagogique - Modèle simplifié de gestion opérationnelle</small>
</div>
""", unsafe_allow_html=True)