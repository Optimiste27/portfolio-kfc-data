"""
Dashboard COS KFC avec authentification
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

# Ajout des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

# Import de l'authentification
try:
    from auth import DashboardAuthenticator, login_page, show_user_profile, check_permissions
    auth_module = True
except ImportError:
    auth_module = False
    st.error("❌ Module d'authentification non trouvé")

# Configuration
st.set_page_config(
    page_title="Dashboard COS KFC Sécurisé",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de l'authentificateur
@st.cache_resource
def get_authenticator():
    """Retourne l'instance d'authentification"""
    users_file = os.path.join(current_dir, "users.json")
    return DashboardAuthenticator(users_file=users_file)

# Vérification d'authentification
def check_auth():
    """Vérifie si l'utilisateur est authentifié"""
    authenticator = get_authenticator()
    
    # Vérifier le token dans la session
    if "auth_token" in st.session_state:
        user_info = authenticator.verify_token(st.session_state.auth_token)
        if user_info:
            st.session_state.user_info = user_info
            return True
    
    return False

# Page de connexion si non authentifié
if not check_auth():
    if auth_module:
        authenticator = get_authenticator()
        login_page(authenticator)
        st.stop()
    else:
        st.error("Système d'authentification indisponible")
        st.stop()

# ===== DASHBOARD PRINCIPAL (utilisateur authentifié) =====

# Titre avec info utilisateur
user_role = st.session_state.user_info.get("role", "viewer")
user_name = st.session_state.user_info.get("full_name", st.session_state.user_info.get("username", "Utilisateur"))

st.title(f"🏪 Dashboard COS KFC - {user_name}")
st.markdown(f"### Rôle: {user_role.upper()} • Surveillance des écarts de coût opérationnel")

# Sidebar avec profil
with st.sidebar:
    show_user_profile()
    
    st.markdown("---")
    st.header("📊 Navigation & Filtres")
    
    # Filtres selon le rôle
    if check_permissions("manager"):
        # Chargement des données (simplifié pour l'exemple)
        @st.cache_data
        def load_data():
            data_dir = os.path.join(root_dir, "data", "raw")
            trans_files = list(Path(data_dir).glob("*transaction*.csv"))
            if trans_files:
                latest = max(trans_files, key=lambda x: x.stat().st_mtime)
                return pd.read_csv(latest)
            return pd.DataFrame()
        
        transactions = load_data()
        
        # Filtres
        if 'restaurant_name' in transactions.columns:
            restaurants = ["Tous"] + sorted(transactions['restaurant_name'].unique().tolist())
            selected_restaurant = st.selectbox("🏪 Restaurant", restaurants)
        else:
            selected_restaurant = "Tous"
        
        if 'product_family' in transactions.columns:
            products = ["Toutes"] + sorted(transactions['product_family'].unique().tolist())
            selected_product = st.selectbox("🍗 Produit", products)
        else:
            selected_product = "Toutes"
        
        # Informations
        st.markdown("---")
        st.markdown("### 📈 Statistiques")
        st.info(f"**{len(transactions):,}** transactions chargées")
    
    else:
        st.info("👁️ Mode consultation - Filtres limités")

# ===== CONTENU SELON LE RÔLE =====

# Onglets communs
tabs = ["📈 Vue d'ensemble", "📊 Analyse"]
if check_permissions("manager"):
    tabs.append("🏪 Gestion restaurants")
if check_permissions("admin"):
    tabs.append("⚙️ Administration")

selected_tab = st.tabs(tabs)

# Tab 1: Vue d'ensemble (tous les rôles)
with selected_tab[0]:
    st.subheader("Vue d'ensemble")
    
    if check_permissions("viewer"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Écart COS moyen", "4.2%", "-0.2%")
        with col2:
            st.metric("Restaurants actifs", "10")
        with col3:
            st.metric("Taux d'anomalies", "12.5%")
        
        # Graphique exemple
        fig, ax = plt.subplots()
        ax.bar(['Jan', 'Fév', 'Mar'], [4.5, 4.2, 4.0])
        ax.axhline(y=4.0, color='r', linestyle='--', label='Cible')
        ax.set_ylabel('Écart COS (%)')
        ax.set_title('Évolution trimestrielle')
        ax.legend()
        st.pyplot(fig)

# Tab 2: Analyse (viewer+)
with selected_tab[1]:
    st.subheader("Analyse détaillée")
    
    if check_permissions("viewer"):
        # Contenu d'analyse
        st.write("""
        ### 📋 Points d'attention
        
        1. **Restaurant Marseille**: Écart de 8.0% - Action requise
        2. **Produit Accompagnements**: Écart moyen de 4.5%
        3. **Semaine 15**: Pic d'anomalies détecté
        
        ### 🎯 Recommandations
        - Audit restaurant Marseille
        - Formation équipes sur procédures
        - Optimisation gestion des stocks
        """)

# Tab 3: Gestion restaurants (manager+)
if check_permissions("manager") and len(tabs) > 2:
    with selected_tab[2]:
        st.subheader("Gestion des restaurants")
        
        # Interface de gestion
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏪 Restaurants critiques")
            st.dataframe({
                'Restaurant': ['Marseille', 'Rennes', 'Nantes'],
                'Écart': ['8.0%', '5.5%', '5.0%'],
                'Statut': ['🔴 Critique', '🟡 Attention', '🟡 Attention']
            })
        
        with col2:
            st.markdown("### 📋 Actions planifiées")
            st.text_area("Plan d'action Marseille:", 
                        "1. Audit complet (15/02)\n2. Formation équipe (20/02)\n3. Suivi rapproché (Mensuel)",
                        height=150)
            
            if st.button("📧 Envoyer plan d'action", use_container_width=True):
                st.success("Plan envoyé au manager Marseille")

# Tab 4: Administration (admin seulement)
if check_permissions("admin") and len(tabs) > 3:
    with selected_tab[3]:
        st.subheader("Administration du système")
        
        authenticator = get_authenticator()
        
        tab_admin1, tab_admin2, tab_admin3 = st.tabs(["👥 Utilisateurs", "🔧 Configuration", "📊 Logs"])
        
        with tab_admin1:
            st.markdown("### Gestion des utilisateurs")
            
            # Liste des utilisateurs
            users = authenticator.get_all_users()
            st.dataframe(pd.DataFrame(users).T)
            
            # Ajout d'utilisateur
            with st.expander("➕ Ajouter un utilisateur"):
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    new_user = st.text_input("Nom d'utilisateur")
                    new_pass = st.text_input("Mot de passe", type="password")
                with col_u2:
                    new_role = st.selectbox("Rôle", ["viewer", "manager", "admin"])
                    new_name = st.text_input("Nom complet")
                
                if st.button("Créer l'utilisateur"):
                    if authenticator.add_user(new_user, new_pass, new_role, new_name):
                        st.success(f"Utilisateur {new_user} créé")
                        st.rerun()
                    else:
                        st.error("Erreur de création")
        
        with tab_admin2:
            st.markdown("### Configuration système")
            st.write("Paramètres du dashboard...")
        
        with tab_admin3:
            st.markdown("### Journal d'activité")
            st.write("Logs des connexions et actions...")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    🔒 Dashboard sécurisé • {user_role.upper()} • {datetime.now().strftime('%d/%m/%Y %H:%M')}
</div>
""", unsafe_allow_html=True)