import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px # Pour des graphiques interactifs
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# --- CHARGEMENT DES ASSETS ---
@st.cache_resource
def load_model():
    model = joblib.load('modele_titanic.pkl')
    scaler = joblib.load('scaler_titanic.pkl')
    return model, scaler

# Vérification de l'existence des fichiers
if os.path.exists('modele_titanic.pkl') and os.path.exists('scaler_titanic.pkl'):
    model, scaler = load_model()
else:
    st.error("Fichiers du modèle manquants ! Veuillez exécuter le notebook d'abord.")
    st.stop()

# --- DESIGN CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- NAVIGATION ---
st.title("🚢 Système Expert de Survie du Titanic")
tab1, tab2, tab3 = st.tabs(["🔮 Prédiction Directe", "📊 Statistiques & EDA", "🧠 Analyse du Modèle"])

# --- ONGLET 1 : PRÉDICTION ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profil du Passager")
        sex = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
        pclass = st.selectbox("Classe de voyage", [1, 2, 3], format_func=lambda x: f"{x}ère Classe" if x==1 else f"{x}ème Classe")
        age = st.slider("Âge", 0, 100, 25)
        fare = st.number_input("Prix du billet (en £)", 0, 512, 32)
        
        with st.expander("Détails familiaux"):
            sibsp = st.number_input("Frères/Sœurs & Conjoint à bord", 0, 8, 0)
            parch = st.number_input("Parents & Enfants à bord", 0, 6, 0)
            embarked = st.selectbox("Port d'embarquement", ["Cherbourg (France)", "Queenstown (Irlande)", "Southampton (UK)"])

    with col2:
        st.subheader("Résultat de l'Analyse")
        
        # Encodage pour le modèle
        sex_val = 1 if sex == "Homme" else 0
        emb_map = {"Cherbourg (France)": 0, "Queenstown (Irlande)": 1, "Southampton (UK)": 2}
        emb_val = emb_map[embarked]
        
        input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, emb_val]], 
                                 columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])
        
        # Prediction
        input_scaled = scaler.transform(input_data)
        if st.button("Calculer les chances de survie"):
            prob = model.predict_proba(input_scaled)[0][1]
            pred = model.predict(input_scaled)[0]
            
            if pred == 1:
                st.balloons()
                st.markdown(f'<div class="result-box" style="background-color: #d4edda; color: #155724;">✅ SURVIE PROBABLE ({prob:.1%})</div>', unsafe_allow_html=True)
                st.write("Le profil de ce passager correspond aux groupes prioritaires lors de l'évacuation (Femmes, enfants ou hautes classes).")
            else:
                st.markdown(f'<div class="result-box" style="background-color: #f8d7da; color: #721c24;">❌ DÉCÈS PROBABLE ({(1-prob):.1%})</div>', unsafe_allow_html=True)
                st.write("Historiquement, ce profil de passager avait peu de chances d'accéder aux canots de sauvetage.")

# --- ONGLET 2 : STATISTIQUES ---
with tab2:
    st.header("Analyse Visuelle des Données Historiques")
    
    # On charge les données réelles pour les graphes dynamiques
    import seaborn as sns
    raw_data = sns.load_dataset('titanic')
    
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(raw_data, x="age", color="survived", nbins=30, title="Distribution des âges selon la survie")
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        fig2 = px.sunburst(raw_data, path=['pclass', 'sex'], values='survived', title="Répartition Survie par Classe et Sexe")
        st.plotly_chart(fig2, use_container_width=True)

    if os.path.exists('graphe/survie_sexe_classe.png'):
        st.image('graphe/survie_sexe_classe.png', caption="Analyse croisée Sexe/Classe issue du Notebook")

# --- ONGLET 3 : COMPRENDRE LE MODÈLE ---
with tab3:
    st.header("Comment l'IA prend-elle sa décision ?")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("""
        Le modèle utilisé est un **Random Forest (Forêt Aléatoire)** optimisé. 
        Il analyse des centaines de scénarios ("Arbres") pour voter la décision finale.
        """)
        
        if os.path.exists('graphe/importance_variables.png'):
            st.image('graphe/importance_variables.png', caption="Poids de chaque critère dans la décision")
            
    with col_b:
        st.write("### Performances validées")
        st.info("Précision du modèle : **~83%**")
        if os.path.exists('graphe/matrice_confusion.png'):
            st.image('graphe/matrice_confusion.png', caption="Matrice de Confusion")

st.markdown("---")
st.caption("Projet Titanic - Analyse Prédictive")
