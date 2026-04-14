# PCOS Predictor - Détection du Syndrome des Ovaires Polykystiques

##  Description du Projet
Application de Machine Learning permettant de prédire la probabilité d’avoir le Syndrome des Ovaires Polykystiques (PCOS) à partir de données cliniques et hormonales.

**Algorithme retenu :** LightGBM  
**Performance :** F1-Score = 0.98 | ROC-AUC = 0.997

##  Structure du Projet
PCOS_Predictor/

├── notebooks/    
├── app.py                
├── requirements.txt
├── data/
├── model/
├── README.md
└── presentation.pptx


##  Fonctionnalités
- Interface utilisateur intuitive avec Streamlit
- Prédiction en temps réel avec probabilité
- Explication des facteurs de risque détectés
- Glossaire médical pour chaque champ

##  Technologies utilisées
- Python
- Pandas, Scikit-learn
- LightGBM
- Streamlit
- Joblib

##  Démarche ML
1. Exploration des données
2. Préprocessing et nettoyage
3. Feature engineering (suppression du data leakage)
4. Comparaison de modèles (LightGBM, Random Forest, Logistic Regression)
5. Sélection et optimisation du meilleur modèle
6. Déploiement via Streamlit

##  Résultats
- Meilleur modèle : **LightGBM**
- F1-Score : 0.9878
- ROC-AUC : 0.9970

## ▶ Comment exécuter l'application ?
```bash
pip install -r requirements.txt
streamlit run app.py
```
## Auteur 
Yasmine Douik

## Licence
Projet réalisé dans le cadre académique.
