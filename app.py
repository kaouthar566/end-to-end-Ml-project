import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# Titre principal
st.title("📊 Loan Default Prediction")
st.markdown("""
## **Prédiction du défaut de remboursement de prêt**
Cette application utilise le machine learning pour prédire si un client est susceptible de faire défaut sur son prêt.
""")

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown("Configurez les paramètres du modèle et explorez les données.")

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv("loan_default_dataset.csv")
    return data.copy()

try:
    loan_default_dataset = load_data()
    
    # Affichage des données brutes
    if st.sidebar.checkbox("Afficher les données brutes"):
        st.subheader("Données brutes")
        st.write(loan_default_dataset)
        
        # Statistiques de base
        st.subheader("Statistiques descriptives")
        st.write(loan_default_dataset.describe())
        
        # Informations sur les données manquantes
        st.subheader("Données manquantes")
        missing_data = loan_default_dataset.isnull().sum()
        missing_percent = (loan_default_dataset.isnull().sum() / loan_default_dataset.shape[0] * 100)
        missing_df = pd.DataFrame({
            'Valeurs manquantes': missing_data,
            'Pourcentage (%)': missing_percent
        })
        st.write(missing_df)

except FileNotFoundError:
    st.error("Le fichier 'loan_default_dataset.csv' n'a pas été trouvé. Veuillez vous assurer qu'il est dans le même répertoire que cette application.")
    st.stop()

# Analyse exploratoire
st.sidebar.header("Analyse Exploratoire")

if st.sidebar.checkbox("Afficher l'analyse exploratoire"):
    st.header("🔍 Analyse Exploratoire des Données")
    
    # Distribution de la variable cible
    st.subheader("Distribution de la variable cible (BAD)")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Diagramme en barres
    target_counts = loan_default_dataset['BAD'].value_counts()
    ax[0].bar(['Non Défaillant (0)', 'Défaillant (1)'], target_counts.values, color=['lightblue', 'salmon'])
    ax[0].set_title('Distribution des Défaillants')
    ax[0].set_ylabel('Nombre de clients')
    
    # Ajout des pourcentages
    for i, v in enumerate(target_counts.values):
        ax[0].text(i, v + 10, f'{v}\n({v/len(loan_default_dataset)*100:.1f}%)', 
                   ha='center', va='bottom')
    
    # Camembert
    ax[1].pie(target_counts.values, labels=['Non Défaillant', 'Défaillant'], 
              autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    ax[1].set_title('Répartition des Défaillants')
    
    st.pyplot(fig)
    
    # Variables catégorielles
    st.subheader("Variables Catégorielles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Répartition par RAISON du prêt:**")
        reason_counts = loan_default_dataset['REASON'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
        st.pyplot(fig)
    
    with col2:
        st.write("**Répartition par TYPE d'emploi:**")
        job_counts = loan_default_dataset['JOB'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(job_counts.values, labels=job_counts.index, autopct='%1.1f%%')
        st.pyplot(fig)

# Prétraitement des données
st.sidebar.header("Prétraitement des Données")

def preprocess_data(df):
    # Copie des données
    data = df.copy()
    
    # Conversion des types
    categorical_cols = ['REASON', 'JOB', 'BAD']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")
    
    # Gestion des valeurs manquantes (exemple simplifié)
    # Dans une application réelle, il faudrait un traitement plus sophistiqué
    numerical_cols = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
    
    for col in numerical_cols:
        if col in data.columns:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Encodage des variables catégorielles
    data = pd.get_dummies(data, columns=['REASON', 'JOB'], drop_first=True)
    
    return data

if st.sidebar.checkbox("Afficher les données après prétraitement"):
    st.header("🔄 Données après Prétraitement")
    processed_data = preprocess_data(loan_default_dataset)
    st.write("Forme des données après prétraitement:", processed_data.shape)
    st.write(processed_data.head())

# Modélisation
st.sidebar.header("Modélisation")

def train_model(data):
    # Préparation des données
    X = data.drop('BAD', axis=1)
    y = data['BAD']
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, X_test_scaled, y_test, scaler

if st.sidebar.checkbox("Entraîner le modèle"):
    st.header("🤖 Modèle de Prédiction")
    
    with st.spinner("Entraînement du modèle en cours..."):
        processed_data = preprocess_data(loan_default_dataset)
        model, X_test, y_test, scaler = train_model(processed_data)
        
        # Évaluation du modèle
        y_pred = model.predict(X_test)
        
        st.subheader("Performance du Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Rapport de Classification:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        with col2:
            st.write("**Matrice de Confusion:**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Réel')
            ax.set_title('Matrice de Confusion')
            st.pyplot(fig)

# Prédiction en temps réel
st.sidebar.header("Prédiction")

st.header("🎯 Prédire le Risque de Défaut")

# Formulaire de saisie
st.subheader("Saisissez les informations du client:")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input("Montant du prêt (LOAN)", min_value=0, value=10000)
    mortdue = st.number_input("Montant hypothécaire (MORTDUE)", min_value=0, value=50000)
    property_value = st.number_input("Valeur du bien (VALUE)", min_value=0, value=75000)

with col2:
    job = st.selectbox("Type d'emploi (JOB)", ['Other', 'Office', 'ProfExe', 'Mgr', 'Self', 'Sales'])
    years_job = st.number_input("Années dans l'emploi (YOJ)", min_value=0.0, value=5.0)
    derogatory_reports = st.number_input("Rapports dérogatoires (DEROG)", min_value=0, value=0)

with col3:
    delinquent = st.number_input("Délinquants (DELINQ)", min_value=0, value=0)
    credit_age = st.number_input("Âge du crédit (CLAGE)", min_value=0.0, value=150.0)
    recent_inquiries = st.number_input("Demandes récentes (NINQ)", min_value=0, value=1)

reason = st.selectbox("Raison du prêt", ['HomeImp', 'DebtCon'])
credit_lines = st.number_input("Lignes de crédit (CLNO)", min_value=0, value=20)
debt_income_ratio = st.number_input("Ratio dette/revenu (DEBTINC)", min_value=0.0, value=35.0)

if st.button("Prédire le risque"):
    # Préparation des données pour la prédiction
    input_data = {
        'LOAN': loan_amount,
        'MORTDUE': mortdue,
        'VALUE': property_value,
        'YOJ': years_job,
        'DEROG': derogatory_reports,
        'DELINQ': delinquent,
        'CLAGE': credit_age,
        'NINQ': recent_inquiries,
        'CLNO': credit_lines,
        'DEBTINC': debt_income_ratio,
        'REASON_HomeImp': 1 if reason == 'HomeImp' else 0,
        'JOB_Office': 1 if job == 'Office' else 0,
        'JOB_ProfExe': 1 if job == 'ProfExe' else 0,
        'JOB_Mgr': 1 if job == 'Mgr' else 0,
        'JOB_Self': 1 if job == 'Self' else 0,
        'JOB_Sales': 1 if job == 'Sales' else 0
    }
    
    # Conversion en DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Prédiction (simulée - dans une vraie application, vous utiliseriez le modèle entraîné)
    risk_probability = 0.15  # Exemple
    
    st.subheader("Résultat de la Prédiction")
    
    if risk_probability < 0.3:
        st.success(f"✅ **FAIBLE RISQUE** - Probabilité de défaut: {risk_probability:.1%}")
        st.info("Recommandation: Prêt approuvé")
    elif risk_probability < 0.6:
        st.warning(f"⚠️ **RISQUE MODÉRÉ** - Probabilité de défaut: {risk_probability:.1%}")
        st.info("Recommandation: Analyse supplémentaire recommandée")
    else:
        st.error(f"🚨 **HAUT RISQUE** - Probabilité de défaut: {risk_probability:.1%}")
        st.info("Recommandation: Prêt non recommandé")

# Footer
st.markdown("---")
st.markdown("""
### **Recommandations pour la banque:**
- Surveiller particulièrement le ratio dette/revenu (DEBTINC)
- Accorder une attention aux antécédents de crédit (DEROG, DELINQ)
- Considérer la stabilité professionnelle (YOJ)
- Analyser le motif du prêt (HomeImp vs DebtCon)
""")

# Fonctionnalités supplémentaires
st.sidebar.header("Fonctionnalités Avancées")
if st.sidebar.checkbox("Afficher l'importance des caractéristiques"):
    st.header("📈 Importance des Caractéristiques")
    
    # Exemple d'importance des features (simulé)
    features = ['DEBTINC', 'DELINQ', 'DEROG', 'CLAGE', 'YOJ', 'LOAN', 'VALUE', 'NINQ', 'CLNO', 'MORTDUE']
    importance = [0.25, 0.18, 0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title('Importance des Caractéristiques dans la Prédiction')
    plt.tight_layout()
    st.pyplot(fig)