"""
Authentification simple pour dashboard COS KFC
"""

import streamlit as st
import json
import os
from datetime import datetime

class SimpleAuth:
    def __init__(self):
        self.users_file = "users.json"
        self._ensure_users_file()
    
    def _ensure_users_file(self):
        """Crée le fichier users.json s'il n'existe pas"""
        if not os.path.exists(self.users_file):
            default_users = {
                "admin": {"password": "admin123", "role": "admin", "name": "Admin"},
                "manager": {"password": "manager123", "role": "manager", "name": "Manager"},
                "viewer": {"password": "viewer123", "role": "viewer", "name": "Viewer"}
            }
            with open(self.users_file, 'w') as f:
                json.dump(default_users, f, indent=2)
    
    def check_login(self, username, password):
        """Vérifie le login"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users and users[username]["password"] == password:
                return True, users[username]
            return False, None
        except:
            return False, None
    
    def get_user_session(self, username, user_data):
        """Crée une session utilisateur"""
        return {
            "username": username,
            "role": user_data["role"],
            "name": user_data["name"],
            "login_time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }


def show_login():
    """Affiche la page de login"""
    st.set_page_config(page_title="Connexion", layout="centered")
    
    st.title("🔐 Dashboard COS KFC")
    st.markdown("---")
    
    auth = SimpleAuth()
    
    username = st.text_input("Nom d'utilisateur", value="admin")
    password = st.text_input("Mot de passe", type="password", value="admin123")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔑 Se connecter", use_container_width=True, type="primary"):
            success, user_data = auth.check_login(username, password)
            if success:
                st.session_state.user = auth.get_user_session(username, user_data)
                st.session_state.logged_in = True
                st.success(f"✅ Connecté en tant que {username}")
                st.rerun()
            else:
                st.error("❌ Identifiants incorrects")
    
    with col2:
        if st.button("👁️ Mode démo", use_container_width=True):
            st.session_state.user = {
                "username": "viewer",
                "role": "viewer",
                "name": "Démo",
                "login_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.logged_in = True
            st.success("✅ Mode démo activé")
            st.rerun()
    
    st.markdown("---")
    st.markdown("**Comptes de test:**")
    st.code("admin / admin123\nmanager / manager123\nviewer / viewer123")
    st.caption("Mots de passe en clair pour la démo")


def is_logged_in():
    """Vérifie si l'utilisateur est connecté"""
    return st.session_state.get("logged_in", False)


def get_current_user():
    """Retourne l'utilisateur courant"""
    return st.session_state.get("user", {})


def logout():
    """Déconnexion"""
    for key in ["logged_in", "user"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def has_permission(required_role):
    """Vérifie les permissions"""
    user = get_current_user()
    role_level = {"viewer": 1, "manager": 2, "admin": 3}
    user_role = user.get("role", "viewer")
    return role_level.get(user_role, 0) >= role_level.get(required_role, 0)



def init_session_state():
    """Initialise l'état de session"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = {}

def require_login():
    """Redirige vers la page de login si non connecté"""
    init_session_state()
    
    if not st.session_state.logged_in:
        show_login()
        st.stop()

def get_user_role():
    """Retourne le rôle de l'utilisateur connecté"""
    return st.session_state.user.get("role", "viewer")

def can_access(required_role="viewer"):
    """Vérifie si l'utilisateur peut accéder à une fonctionnalité"""
    user_role = get_user_role()
    role_level = {"viewer": 1, "manager": 2, "admin": 3}
    return role_level.get(user_role, 0) >= role_level.get(required_role, 0)