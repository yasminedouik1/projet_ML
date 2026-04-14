import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="PCOS Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f4fb; }
    .stApp { background-color: #f8f4fb; }

    .header-box {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-box h1 { color: white; font-size: 2.2rem; margin: 0; }
    .header-box p  { color: rgba(255,255,255,0.88); margin: 0.5rem 0 0; font-size: 1rem; }

    .result-positive {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 2px solid #f87171;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 2px solid #4ade80;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-positive h2 { color: #dc2626; font-size: 1.8rem; }
    .result-negative h2 { color: #16a34a; font-size: 1.8rem; }

    .section-title {
        color: #7c3aed;
        font-size: 1.05rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e9d5ff;
    }
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 1.5rem;
    }
    div[data-testid="stSidebar"] { background-color: #faf5ff; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model    = joblib.load(os.path.join(base, "./model/pcos_model.pkl"))
    features = joblib.load(os.path.join(base, "./model/features.pkl"))
    return model, features

model, FEATURES = load_model()

st.markdown("""
<div class="header-box">
  <h1>🩺 PCOS Prediction App</h1>
  <p>Détection du Syndrome des Ovaires Polykystiques par Machine Learning (LightGBM · F1=0.99 · AUC=0.997)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 À propos du modèle")
    st.markdown("""
    **Algorithme** : LightGBM (Gradient Boosting)  
    **Dataset** : 2 000 patientes · 44 features  
    **Validation** : StratifiedKFold (10 folds)
    """)
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("F1-Score", "0.9878")
    col2.metric("ROC-AUC",  "0.9970")
    st.divider()
    st.markdown("**Features utilisées** :")
    for f in FEATURES:
        st.markdown(f"- `{f}`")

st.markdown("### 📋 Entrez les données de la patiente")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown('<div class="section-title">👤 Informations générales</div>', unsafe_allow_html=True)
    age         = st.number_input("Âge (ans)",             min_value=10,   max_value=60,   value=25,  step=1)
    bmi         = st.number_input("BMI",                   min_value=10.0, max_value=60.0, value=22.0, step=0.1)
    pregnant    = st.selectbox("Grossesse actuelle",       options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    beta_hcg    = st.number_input("II beta-HCG (mIU/mL)", min_value=0.0,  max_value=500.0, value=1.0, step=0.1)

with col_b:
    st.markdown('<div class="section-title">🔬 Hormones & Biologie</div>', unsafe_allow_html=True)
    amh         = st.number_input("AMH (ng/mL)",           min_value=0.0,  max_value=20.0, value=2.5,  step=0.1)
    lh          = st.number_input("LH (mIU/mL)",           min_value=0.0,  max_value=50.0, value=4.5,  step=0.1)
    fsh_lh      = st.number_input("FSH/LH Ratio",          min_value=0.0,  max_value=10.0, value=1.1,  step=0.01)
    waist_hip   = st.number_input("Waist:Hip Ratio",       min_value=0.5,  max_value=1.5,  value=0.8,  step=0.01)
    endometrium = st.number_input("Endometrium (mm)",      min_value=0.0,  max_value=30.0, value=7.0,  step=0.1)

with col_c:
    st.markdown('<div class="section-title">🩹 Symptômes cliniques</div>', unsafe_allow_html=True)
# APRÈS (correct — valeurs réelles du dataset)
    cycle = st.selectbox("Cycle menstruel", options=[4, 2], format_func=lambda x: "Régulier (R)" if x==4 else "Irrégulier (I)")    weight_gain = st.selectbox("Prise de poids",           options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    hair_growth = st.selectbox("Pilosité excessive",       options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    skin_dark   = st.selectbox("Assombrissement peau",     options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    hair_loss   = st.selectbox("Perte de cheveux",         options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    pimples     = st.selectbox("Acné / Boutons",           options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    fast_food   = st.selectbox("Consommation fast food",   options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")

st.markdown("---")
predict_btn = st.button("🔍 Lancer la prédiction", use_container_width=True, type="primary")

if predict_btn:
    input_dict = {
        'Weight gain(Y/N)':       weight_gain,
        'hair growth(Y/N)':       hair_growth,
        'Skin darkening (Y/N)':   skin_dark,
        'Pimples(Y/N)':           pimples,
        'Hair loss(Y/N)':         hair_loss,
        'AMH(ng/mL)':             amh,
        'Cycle(R/I)':             cycle,
        'BMI':                    bmi,
        'Waist:Hip Ratio':        waist_hip,
        'LH(mIU/mL)':             lh,
        'FSH/LH':                 fsh_lh,
        'II    beta-HCG(mIU/mL)': beta_hcg,
        'Age (yrs)':              age,
        'Pregnant(Y/N)':          pregnant,
        'Endometrium (mm)':       endometrium,
        'Fast food (Y/N)':        fast_food,
    }

    input_df   = pd.DataFrame([input_dict])[FEATURES]
    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    prob_pcos  = proba[1] * 100

    st.markdown("### 🎯 Résultat de la prédiction")
    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
              <h2>⚠️ PCOS Détecté</h2>
              <p style="font-size:1.1rem; color:#b91c1c;">
                Probabilité : <strong>{prob_pcos:.1f}%</strong>
              </p>
              <p style="color:#7f1d1d; font-size:0.9rem;">
                Une consultation médicale spécialisée est fortement recommandée.
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
              <h2>✅ Pas de PCOS détecté</h2>
              <p style="font-size:1.1rem; color:#15803d;">
                Probabilité PCOS : <strong>{prob_pcos:.1f}%</strong>
              </p>
              <p style="color:#14532d; font-size:0.9rem;">
                Les indicateurs ne suggèrent pas de PCOS. Suivi médical régulier conseillé.
              </p>
            </div>
            """, unsafe_allow_html=True)

    with gauge_col:
        st.markdown("**Probabilités par classe**")
        st.metric("🔴 PCOS",     f"{prob_pcos:.1f}%")
        st.metric("🟢 Non-PCOS", f"{100 - prob_pcos:.1f}%")
        st.progress(int(prob_pcos))

        st.markdown("**Facteurs de risque détectés**")
        risk_factors = []
        if weight_gain: risk_factors.append("✔ Prise de poids")
        if hair_growth: risk_factors.append("✔ Pilosité excessive")
        if skin_dark:   risk_factors.append("✔ Assombrissement de la peau")
        if hair_loss:   risk_factors.append("✔ Perte de cheveux")
        if pimples:     risk_factors.append("✔ Acné")
        if fast_food:   risk_factors.append("✔ Consommation fast food")
        if lh > 10:     risk_factors.append("✔ LH élevée")
        if amh > 5:     risk_factors.append("✔ AMH élevée")
        if cycle == 2:  risk_factors.append("✔ Cycle irrégulier")
        if risk_factors:
            for r in risk_factors:
                st.markdown(r)
        else:
            st.markdown("*Aucun facteur de risque majeur détecté*")

    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Avertissement médical</strong> : Cette application est un outil d'aide à la décision
      basé sur un modèle de Machine Learning. Elle ne remplace en aucun cas un diagnostic médical
      professionnel. Consultez un gynécologue pour tout diagnostic.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 📖 Explication des champs")

with st.expander("👤 Informations générales — Cliquez pour afficher", expanded=False):
    st.markdown("""
    | Champ | Signification | Valeurs normales |
    |-------|--------------|-----------------|
    | **Âge (ans)** | Âge de la patiente. Le PCOS touche principalement les femmes en âge de procréer (15–45 ans). | 15 – 45 ans |
    | **BMI** | Indice de Masse Corporelle (poids kg / taille m²). Un BMI élevé est un facteur de risque du PCOS et de la résistance à l'insuline. | 18.5 – 24.9 (normal) |
    | **Grossesse actuelle** | Indique si la patiente est actuellement enceinte. | Oui / Non |
    | **II beta-HCG (mIU/mL)** | Gonadotrophine chorionique humaine — hormone produite pendant la grossesse, aussi utilisée pour évaluer l'activité ovarienne. | < 5 mIU/mL (non enceinte) |
    """)

with st.expander("🔬 Hormones & Biologie — Cliquez pour afficher", expanded=False):
    st.markdown("""
    | Champ | Signification | Valeurs normales |
    |-------|--------------|-----------------|
    | **AMH (ng/mL)** | Anti-Müllerian Hormone — mesure la réserve ovarienne. Un taux élevé indique souvent un excès de follicules, signe caractéristique du PCOS. | 1.0 – 3.5 ng/mL |
    | **LH (mIU/mL)** | Hormone Lutéinisante — déclenche l'ovulation. Dans le PCOS, le taux de LH est souvent anormalement élevé. | 2.4 – 12.6 mIU/mL |
    | **FSH/LH Ratio** | Rapport entre FSH et LH. Un ratio < 1 (LH > FSH) est fortement associé au PCOS. | > 1 (normal) |
    | **Waist:Hip Ratio** | Rapport tour de taille / tour de hanches. Un ratio élevé indique une obésité abdominale, liée au PCOS. | < 0.85 (femme) |
    | **Endometrium (mm)** | Épaisseur de la muqueuse utérine mesurée par échographie. Une épaisseur anormale peut accompagner le PCOS. | 2 – 12 mm selon la phase |
    """)

with st.expander("🩹 Symptômes cliniques — Cliquez pour afficher", expanded=False):
    st.markdown("""
    | Champ | Signification | Lien avec PCOS |
    |-------|--------------|----------------|
    | **Cycle menstruel** | Régulier ou Irrégulier. Un cycle irrégulier est l'un des critères diagnostiques principaux du PCOS. | ⚠️ Critère majeur |
    | **Prise de poids** | Prise de poids inexpliquée ou rapide. Fréquente dans le PCOS à cause de la résistance à l'insuline. | ⚠️ Fréquent |
    | **Pilosité excessive** | Hirsutisme — poils en excès sur le visage, le ventre ou le dos. Causé par l'excès d'androgènes. | ⚠️ Critère majeur |
    | **Assombrissement de la peau** | Acanthosis nigricans — taches sombres sur le cou, les aisselles. Signe de résistance à l'insuline. | ⚠️ Indicateur |
    | **Perte de cheveux** | Alopécie androgénique — chute de cheveux liée à l'excès d'hormones mâles. | ⚠️ Fréquent |
    | **Acné / Boutons** | Acné hormonale persistante, souvent sur le menton et la mâchoire. Liée à l'excès d'androgènes. | ⚠️ Fréquent |
    | **Consommation fast food** | Une alimentation riche en graisses et sucres favorise la résistance à l'insuline, facteur aggravant du PCOS. | ⚠️ Facteur aggravant |
    """)

with st.expander("ℹ️ Comment interpréter le résultat ?", expanded=False):
    st.markdown("""
    Le modèle **LightGBM** calcule une **probabilité** entre 0% et 100% d'avoir le PCOS.

    | Probabilité | Interprétation | Action recommandée |
    |-------------|---------------|-------------------|
    | **< 30%** | Risque faible | Suivi médical régulier |
    | **30% – 60%** | Risque modéré | Consultation gynécologique conseillée |
    | **> 60%** | Risque élevé | Consultation spécialisée urgente recommandée |


    """)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "PCOS Predictor - Par Yasmine Douik - 2026"
    "</p>",
    unsafe_allow_html=True
)