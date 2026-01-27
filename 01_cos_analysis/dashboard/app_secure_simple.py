"""
Dashboard COS KFC avec authentification - Version simplifiée
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import base64
from pathlib import Path

# === CONFIGURATION DES CHEMINS ===
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

# Import de l'authentification
try:
    from auth_corrected import (
        DashboardAuthenticator, 
        login_page, 
        check_auth, 
        show_user_profile,
        check_permissions
    )
    auth_module = True
except ImportError as e:
    auth_module = False
    st.error(f"❌ Erreur d'importation: {e}")

# === AUTHENTIFICATION ===
if not auth_module:
    st.error("Module d'authentification non disponible")
    st.stop()

if not check_auth():
    authenticator = DashboardAuthenticator()
    login_page(authenticator)
    st.stop()

# === DASHBOARD PRINCIPAL ===
user_info = st.session_state.user_info
user_role = user_info.get("role", "viewer")
user_name = user_info.get("full_name", user_info.get("username", "Utilisateur"))

# Configuration de la page
st.set_page_config(
    page_title=f"Dashboard COS KFC - {user_name}",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title(f"🏪 Dashboard COS KFC")
st.markdown(f"#### Connecté en tant que **{user_name}** • Rôle: **{user_role.upper()}** • {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Sidebar avec profil et navigation
with st.sidebar:
    show_user_profile()
    
    st.markdown("---")
    st.header("📊 Navigation")
    
    # Options de menu selon le rôle
    menu_items = ["📈 Vue d'ensemble", "📊 Analyse"]
    
    if user_role in ["manager", "admin"]:
        menu_items.append("🏪 Gestion restaurants")
    
    if user_role == "admin":
        menu_items.append("⚙️ Administration")
    
    selected_tab = st.radio("Menu", menu_items)

# === FONCTIONS DE CHARGEMENT DES DONNÉES ===
@st.cache_data
def load_sample_data():
    """Charge des données d'exemple"""
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'restaurant': np.random.choice(['KFC Paris', 'KFC Lyon', 'KFC Marseille', 'KFC Toulouse'], n),
        'product': np.random.choice(['Poulet', 'Accompagnements', 'Boissons', 'Desserts'], n),
        'revenue': np.random.uniform(100, 1000, n),
        'gap_percentage': np.random.uniform(0, 10, n),
        'anomaly': np.random.choice([True, False], n, p=[0.2, 0.8])
    })

# === CONTENU DES ONGLETS ===
if selected_tab == "📈 Vue d'ensemble":
    st.header("📈 Vue d'ensemble")
    
    # Charger les données
    data = load_sample_data()
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_gap = data['gap_percentage'].mean()
        st.metric("Écart COS moyen", f"{avg_gap:.2f}%", delta="-0.5%")
    
    with col2:
        total_revenue = data['revenue'].sum()
        st.metric("Chiffre d'affaires", f"{total_revenue:,.0f} €")
    
    with col3:
        anomaly_rate = data['anomaly'].mean() * 100
        st.metric("Taux d'anomalies", f"{anomaly_rate:.1f}%")
    
    with col4:
        unique_rest = data['restaurant'].nunique()
        st.metric("Restaurants", unique_rest)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution des écarts")
        fig, ax = plt.subplots()
        data['gap_percentage'].hist(bins=20, ax=ax, alpha=0.7, color='skyblue')
        ax.axvline(x=4.0, color='red', linestyle='--', label='Cible (4.0%)')
        ax.set_xlabel('Écart COS (%)')
        ax.set_ylabel('Fréquence')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Performance par restaurant")
        rest_stats = data.groupby('restaurant')['gap_percentage'].mean().sort_values()
        
        fig, ax = plt.subplots()
        colors = ['green' if x < 4 else 'orange' if x < 6 else 'red' for x in rest_stats.values]
        ax.barh(rest_stats.index, rest_stats.values, color=colors)
        ax.set_xlabel('Écart moyen (%)')
        ax.axvline(x=4.0, color='red', linestyle='--', alpha=0.5)
        st.pyplot(fig)

elif selected_tab == "📊 Analyse":
    st.header("📊 Analyse détaillée")
    
    st.markdown("""
    ### 🎯 Points d'attention
    
    **1. Restaurant Marseille Vieux Port** (Écart: 8.0%)
    - Audit recommandé avant le 15/02/2024
    - Formation équipes requise
    - Suivi hebdomadaire
    
    **2. Produit Accompagnements** (Écart moyen: 4.5%)
    - Optimisation des portions
    - Révision des procédures de préparation
    - Analyse des coûts matières
    
    **3. Semaine 15 - Pic d'anomalies**
    - Investigation en cours
    - Actions correctives planifiées
    
    ### 📋 Recommandations opérationnelles
    1. Audit des 3 restaurants avec plus d'écarts
    2. Programme de formation standardisé
    3. Revue des procédures de contrôle qualité
    4. Optimisation de la gestion des stocks
    """)

elif selected_tab == "🏪 Gestion restaurants" and user_role in ["manager", "admin"]:
    st.header("🏪 Gestion des restaurants")
    
    tab1, tab2 = st.tabs(["📋 Liste", "📈 Performances"])
    
    with tab1:
        st.subheader("Restaurants suivis")
        
        restaurants = [
            {"Nom": "KFC Marseille Vieux Port", "Écart": "8.0%", "Statut": "🔴 Critique", "Manager": "Jean Dupont"},
            {"Nom": "KFC Rennes Centre", "Écart": "5.5%", "Statut": "🟡 Attention", "Manager": "Marie Martin"},
            {"Nom": "KFC Nantes Erdre", "Écart": "5.0%", "Statut": "🟡 Attention", "Manager": "Pierre Bernard"},
            {"Nom": "KFC Paris Champs", "Écart": "3.8%", "Statut": "✅ Bon", "Manager": "Sophie Petit"},
            {"Nom": "KFC Lyon Part-Dieu", "Écart": "3.5%", "Statut": "✅ Bon", "Manager": "Luc Dubois"},
        ]
        
        st.dataframe(restaurants, use_container_width=True)
        
        # Actions
        with st.expander("➕ Ajouter une action", expanded=False):
            action = st.text_area("Description de l'action:")
            deadline = st.date_input("Date limite")
            
            if st.button("💾 Enregistrer l'action"):
                st.success("Action enregistrée")
    
    with tab2:
        st.subheader("Analyse de performance")
        
        # Graphique
        fig, ax = plt.subplots()
        rest_names = [r["Nom"][:10] + "..." for r in restaurants]
        gaps = [float(r["Écart"].replace("%", "")) for r in restaurants]
        
        bars = ax.bar(rest_names, gaps, color=['red', 'orange', 'orange', 'green', 'green'])
        ax.set_ylabel('Écart COS (%)')
        ax.set_title('Performance des restaurants')
        ax.axhline(y=4.0, color='r', linestyle='--', label='Cible')
        ax.legend()
        
        # Ajouter les valeurs
        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{gap}%', ha='center', va='bottom')
        
        st.pyplot(fig)

elif selected_tab == "⚙️ Administration" and user_role == "admin":
    st.header("⚙️ Administration")
    
    from auth_corrected import DashboardAuthenticator
    authenticator = DashboardAuthenticator()
    
    tab1, tab2 = st.tabs(["👥 Utilisateurs", "🔧 Système"])
    
    with tab1:
        st.subheader("Gestion des utilisateurs")
        
        # Liste des utilisateurs
        users = authenticator.get_all_users()
        st.dataframe(pd.DataFrame(users).T)
        
        # Ajouter un utilisateur
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_user = st.text_input("Nom d'utilisateur")
                new_pass = st.text_input("Mot de passe", type="password")
            
            with col2:
                new_role = st.selectbox("Rôle", ["viewer", "manager", "admin"])
                new_name = st.text_input("Nom complet")
            
            if st.form_submit_button("➕ Ajouter l'utilisateur"):
                # Note: Dans la version complète, il faudrait implémenter add_user
                st.success(f"Utilisateur {new_user} ajouté (fonctionnalité à implémenter)")
    
    with tab2:
        st.subheader("Configuration système")
        st.write("Paramètres du dashboard...")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    🔒 Dashboard COS KFC Sécurisé • {user_role.upper()} • Session active
</div>
""", unsafe_allow_html=True)