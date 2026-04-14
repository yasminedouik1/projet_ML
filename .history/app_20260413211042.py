import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 5px solid #7c3aed;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    .metric-card h4 { margin: 0 0 0.2rem; color: #6b7280; font-size: 0.85rem; text-transform: uppercase; }
    .metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 700; color: #1f2937; }

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

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model    = joblib.load(os.path.join(base, "pcos_model.pkl"))
    features = joblib.load(os.path.join(base, "features.pkl"))
    return model, features

model, FEATURES = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🩺 PCOS Prediction App</h1>
  <p>Détection du Syndrome des Ovaires Polykystiques par Machine Learning (LightGBM · F1=0.99 · AUC=0.997)</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — model info ──────────────────────────────────────────────────────
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

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### 📋 Entrez les données de la patiente")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown('<div class="section-title">🔬 Hormones & Biologie</div>', unsafe_allow_html=True)
    amh      = st.number_input("AMH (ng/mL)",       min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    fsh      = st.number_input("FSH (mIU/mL)",      min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    lh       = st.number_input("LH (mIU/mL)",       min_value=0.0, max_value=50.0, value=4.5, step=0.1)
    fsh_lh   = st.number_input("FSH/LH Ratio",      min_value=0.0, max_value=10.0, value=1.1, step=0.01)
    rbs      = st.number_input("RBS (mg/dl)",        min_value=50.0, max_value=400.0, value=90.0, step=1.0)

with col_b:
    st.markdown('<div class="section-title">📏 Morphologie & Signes vitaux</div>', unsafe_allow_html=True)
    bmi           = st.number_input("BMI",                min_value=10.0, max_value=60.0, value=22.0, step=0.1)
    waist_hip     = st.number_input("Waist:Hip Ratio",    min_value=0.5,  max_value=1.5,  value=0.8,  step=0.01)
    endometrium   = st.number_input("Endometrium (mm)",   min_value=0.0,  max_value=30.0, value=7.0,  step=0.1)
    bp_sys        = st.number_input("BP Systolique (mmHg)", min_value=80,  max_value=200, value=110, step=1)
    bp_dia        = st.number_input("BP Diastolique (mmHg)", min_value=50, max_value=130, value=70,  step=1)

with col_c:
    st.markdown('<div class="section-title">🩹 Symptômes cliniques</div>', unsafe_allow_html=True)
    cycle         = st.selectbox("Cycle menstruel",     options=[1, 2], format_func=lambda x: "Régulier (R)" if x==1 else "Irrégulier (I)")
    weight_gain   = st.selectbox("Prise de poids",      options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    hair_growth   = st.selectbox("Pilosité excessive",  options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    skin_dark     = st.selectbox("Assombrissement peau",options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    hair_loss     = st.selectbox("Perte de cheveux",    options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
    pimples       = st.selectbox("Acné / Boutons",      options=[0, 1], format_func=lambda x: "Non" if x==0 else "Oui")

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Lancer la prédiction", use_container_width=True, type="primary")

if predict_btn:
    input_dict = {
        'Weight gain(Y/N)':      weight_gain,
        'hair growth(Y/N)':      hair_growth,
        'Skin darkening (Y/N)':  skin_dark,
        'Hair loss(Y/N)':        hair_loss,
        'Pimples(Y/N)':          pimples,
        'AMH(ng/mL)':            amh,
        'FSH/LH':                fsh_lh,
        'Waist:Hip Ratio':       waist_hip,
        'LH(mIU/mL)':            lh,
        'BMI':                   bmi,
        'Cycle(R/I)':            cycle,
        'RBS(mg/dl)':            rbs,
        'BP _Systolic (mmHg)':   bp_sys,
        'BP _Diastolic (mmHg)':  bp_dia,
        'Endometrium (mm)':      endometrium,
        'FSH(mIU/mL)':           fsh,
    }

    input_df = pd.DataFrame([input_dict])[FEATURES]

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
        st.metric("🟢 Non-PCOS", f"{100-prob_pcos:.1f}%")
        st.progress(int(prob_pcos))

        st.markdown("**Facteurs de risque détectés**")
        risk_factors = []
        if weight_gain: risk_factors.append("✔ Prise de poids")
        if hair_growth: risk_factors.append("✔ Pilosité excessive")
        if skin_dark:   risk_factors.append("✔ Assombrissement de la peau")
        if hair_loss:   risk_factors.append("✔ Perte de cheveux")
        if pimples:     risk_factors.append("✔ Acné")
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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "PCOS Predictor · LightGBM · Projet ML · 2025"
    "</p>",
    unsafe_allow_html=True
)
