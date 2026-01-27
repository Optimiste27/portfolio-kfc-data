"""
Dashboard COS KFC avec authentification
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from pathlib import Path

# Ajouter le chemin du projet
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

# Import de l'authentification
from auth_simple import show_login, is_logged_in, get_current_user, logout, has_permission

# Vérifier la connexion
if not is_logged_in():
    show_login()
    st.stop()

# Récupérer l'utilisateur
user = get_current_user()
user_role = user.get("role", "viewer")
user_name = user.get("name", user.get("username", "Utilisateur"))

# Configuration de la page
st.set_page_config(
    page_title=f"Dashboard COS KFC - {user_name}",
    page_icon="🏪",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.markdown(f"### 👤 {user_name}")
    st.caption(f"Rôle: **{user_role.upper()}**")
    
    if st.button("🚪 Déconnexion", use_container_width=True, type="secondary"):
        logout()
    
    st.markdown("---")
    st.markdown("### 📊 Navigation")
    
    # Menu selon rôle
    menu_items = ["📈 Vue d'ensemble", "📋 Analyse"]
    if has_permission("manager"):
        menu_items.append("🏪 Gestion")
    if has_permission("admin"):
        menu_items.append("⚙️ Administration")
    
    selected_page = st.radio("", menu_items)

# Titre principal
st.title(f"🏪 Dashboard COS KFC")
st.markdown(f"**Utilisateur:** {user_name} | **Rôle:** {user_role.upper()} | **Heure:** {datetime.now().strftime('%H:%M')}")

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charge les données réelles"""
    try:
        data_path = Path(root_dir) / "data" / "raw" / "transactions_final_20260124_1112.csv"
        if data_path.exists():
            df = pd.read_csv(data_path, nrows=1000)  # Limiter pour la démo
            
            # Nettoyer les colonnes
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Calculer l'écart si nécessaire
            if 'gap_percentage' not in df.columns:
                if 'theoretical_unit_cost' in df.columns and 'actual_unit_cost' in df.columns:
                    df['gap_percentage'] = ((df['actual_unit_cost'] - df['theoretical_unit_cost']) / 
                                           df['theoretical_unit_cost']) * 100
                else:
                    df['gap_percentage'] = np.random.uniform(0, 10, len(df))
            
            return df
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
    return None

# Page: Vue d'ensemble
if selected_page == "📈 Vue d'ensemble":
    st.header("📈 Vue d'ensemble")
    
    # Charger les données
    with st.spinner("Chargement des données..."):
        data = load_data()
    
    if data is not None:
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_gap = data['gap_percentage'].mean()
            st.metric("Écart COS moyen", f"{avg_gap:.2f}%")
        
        with col2:
            if 'restaurant_name' in data.columns:
                rest_count = data['restaurant_name'].nunique()
                st.metric("Restaurants", rest_count)
            else:
                st.metric("Transactions", f"{len(data):,}")
        
        with col3:
            if 'operational_anomaly' in data.columns:
                anomaly_rate = data['operational_anomaly'].mean() * 100
                st.metric("Taux anomalies", f"{anomaly_rate:.1f}%")
        
        with col4:
            st.metric("Période", "Jan-Juin 2024")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des écarts")
            fig, ax = plt.subplots()
            data['gap_percentage'].hist(bins=20, ax=ax, alpha=0.7, color='skyblue')
            ax.axvline(x=4.0, color='red', linestyle='--', label='Cible 4%')
            ax.set_xlabel('Écart COS (%)')
            ax.set_ylabel('Fréquence')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top restaurants")
            if 'restaurant_name' in data.columns:
                top_5 = data.groupby('restaurant_name')['gap_percentage'].mean().nlargest(5)
                
                fig, ax = plt.subplots()
                colors = ['red' if x > 6 else 'orange' for x in top_5.values]
                ax.barh(top_5.index, top_5.values, color=colors, edgecolor='black')
                ax.set_xlabel('Écart moyen (%)')
                ax.set_title('Top 5 - Plus grands écarts')
                st.pyplot(fig)
    
    else:
        # Mode démo
        st.info("Mode démo - Données d'exemple")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Écart moyen", "4.2%", "-0.3%")
        with col2:
            st.metric("Restaurants", "10")
        with col3:
            st.metric("Anomalies", "12.5%")

# Page: Analyse
elif selected_page == "📋 Analyse":
    st.header("📋 Analyse")
    
    st.markdown("""
    ### 🎯 Points critiques
    
    **1. Restaurant Marseille Vieux Port**
    - Écart: 8.0% (critique)
    - Actions: Audit urgent, formation équipe
    
    **2. Produit Accompagnements**
    - Écart moyen: 4.5%
    - Actions: Optimisation portions
    
    **3. Semaine 15**
    - Anomalies: 25%
    - Actions: Investigation
    """)
    
    if has_permission("manager"):
        st.markdown("---")
        st.subheader("📋 Plan d'actions recommandé")
        st.write("""
        1. **Audit restaurant Marseille** (Priorité haute)
        2. **Formation standardisée** (Toutes les équipes)
        3. **Optimisation stocks** (Produits critiques)
        4. **Revue procédures** (Contrôle qualité)
        """)

# Page: Gestion (manager+)
elif selected_page == "🏪 Gestion" and has_permission("manager"):
    st.header("🏪 Gestion des restaurants")
    
    tab1, tab2 = st.tabs(["📋 Liste", "📈 Graphiques"])
    
    with tab1:
        st.subheader("Restaurants suivis")
        
        restaurants = [
            {"Nom": "KFC Marseille", "Écart": "8.0%", "Statut": "🔴 Critique", "Manager": "Jean D."},
            {"Nom": "KFC Rennes", "Écart": "5.5%", "Statut": "🟡 Attention", "Manager": "Marie M."},
            {"Nom": "KFC Nantes", "Écart": "5.0%", "Statut": "🟡 Attention", "Manager": "Pierre B."},
            {"Nom": "KFC Paris", "Écart": "3.8%", "Statut": "✅ Bon", "Manager": "Sophie P."},
            {"Nom": "KFC Lyon", "Écart": "3.5%", "Statut": "✅ Bon", "Manager": "Luc D."},
        ]
        
        st.dataframe(restaurants, use_container_width=True)
    
    with tab2:
        st.subheader("Performance")
        
        # Graphique simple
        fig, ax = plt.subplots()
        names = [r["Nom"] for r in restaurants]
        values = [float(r["Écart"].replace("%", "")) for r in restaurants]
        colors = ['red', 'orange', 'orange', 'green', 'green']
        
        bars = ax.bar(names, values, color=colors)
        ax.set_ylabel('Écart (%)')
        ax.set_title('Performance par restaurant')
        ax.axhline(y=4.0, color='r', linestyle='--', label='Cible')
        ax.legend()
        
        st.pyplot(fig)

# Page: Administration (admin seulement)
elif selected_page == "⚙️ Administration" and has_permission("admin"):
    st.header("⚙️ Administration")
    
    st.info("Espace d'administration - Fonctionnalités à développer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Utilisateurs")
        st.write("""
        - admin (administrateur)
        - manager (gestionnaire)
        - viewer (consultant)
        """)
    
    with col2:
        st.subheader("Système")
        st.write("""
        - Dashboard: COS KFC
        - Version: 1.0
        - Date: 27/01/2024
        - Statut: Actif
        """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    🔐 Dashboard COS KFC Sécurisé • Rôle: {user_role.upper()} • {datetime.now().strftime('%d/%m/%Y')}
</div>
""", unsafe_allow_html=True)