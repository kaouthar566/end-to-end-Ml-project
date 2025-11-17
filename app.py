import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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

def preprocess_data(df, fit_mode=True, feature_columns=None):
    # Copie des données
    data = df.copy()
    
    # Conversion des types
    categorical_cols = ['REASON', 'JOB', 'BAD']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")
    
    # Gestion des valeurs manquantes (exemple simplifié)
    numerical_cols = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
    
    for col in numerical_cols:
        if col in data.columns:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Encodage des variables catégorielles
    if fit_mode:
        # Mode entraînement - créer les dummy variables
        data = pd.get_dummies(data, columns=['REASON', 'JOB'], drop_first=True)
        # Sauvegarder les colonnes pour la prédiction
        st.session_state['feature_columns'] = data.drop('BAD', axis=1).columns.tolist()
    else:
        # Mode prédiction - utiliser les mêmes colonnes que lors de l'entraînement
        data = pd.get_dummies(data, columns=['REASON', 'JOB'], drop_first=False)
        
        # S'assurer que nous avons toutes les colonnes nécessaires
        if feature_columns is not None:
            # Ajouter les colonnes manquantes avec des valeurs 0
            for col in feature_columns:
                if col not in data.columns:
                    data[col] = 0
            
            # Réorganiser les colonnes dans le même ordre
            data = data[feature_columns + ['BAD'] if 'BAD' in data.columns else feature_columns]
    
    return data

if st.sidebar.checkbox("Afficher les données après prétraitement"):
    st.header("🔄 Données après Prétraitement")
    processed_data = preprocess_data(loan_default_dataset)
    st.write("Forme des données après prétraitement:", processed_data.shape)
    st.write(processed_data.head())

# Modélisation
st.sidebar.header("Modélisation")

# Sélection du modèle
model_choice = st.sidebar.selectbox(
    "Choisissez le modèle de prédiction:",
    ["Forêt Aléatoire", "Régression Logistique", "Arbre de Décision"],
    index=0
)

# Paramètres du modèle selon le choix
if model_choice == "Arbre de Décision":
    max_depth = st.sidebar.slider("Profondeur maximale de l'arbre", 1, 20, 10)
    min_samples_split = st.sidebar.slider("Échantillons minimum pour diviser", 2, 20, 2)

def train_model(data, model_type="Forêt Aléatoire"):
    # Préparation des données
    X = data.drop('BAD', axis=1)
    y = data['BAD']
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sélection et entraînement du modèle
    if model_type == "Forêt Aléatoire":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Régression Logistique":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "Arbre de Décision":
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    return model, X_test_scaled, y_test, scaler, X_train_scaled, y_train, X.columns.tolist()

if st.sidebar.checkbox("Entraîner le modèle"):
    st.header("🤖 Modèle de Prédiction")
    st.write(f"**Modèle sélectionné :** {model_choice}")
    
    with st.spinner(f"Entraînement du modèle {model_choice} en cours..."):
        processed_data = preprocess_data(loan_default_dataset, fit_mode=True)
        model, X_test, y_test, scaler, X_train, y_train, feature_names = train_model(processed_data, model_choice)
        
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
        
        # Stocker le modèle et les informations dans la session
        st.session_state['trained_model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['model_trained'] = True
        st.session_state['model_type'] = model_choice
        st.session_state['feature_names'] = feature_names
        
        st.success("Modèle entraîné avec succès!")

# Prédiction en temps réel
st.sidebar.header("Prédiction")

st.header("🎯 Prédire le Risque de Défaut")

# Afficher le modèle sélectionné
if 'model_trained' in st.session_state and st.session_state['model_trained']:
    st.write(f"**Modèle utilisé pour la prédiction :** {st.session_state['model_type']}")
else:
    st.warning("Veuillez d'abord entraîner un modèle en cochant 'Entraîner le modèle' dans la sidebar.")

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
    # Vérifier si un modèle est entraîné
    if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
        st.error("Veuillez d'abord entraîner un modèle en cochant 'Entraîner le modèle' dans la sidebar.")
    else:
        try:
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
                'REASON': reason,
                'JOB': job
            }
            
            # Conversion en DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Prétraiter les données de la même manière que lors de l'entraînement
            input_processed = preprocess_data(input_df, fit_mode=False, feature_columns=st.session_state['feature_names'])
            
            # Supprimer BAD s'il existe (pour la prédiction)
            if 'BAD' in input_processed.columns:
                input_processed = input_processed.drop('BAD', axis=1)
            
            # S'assurer que toutes les colonnes sont présentes et dans le bon ordre
            missing_cols = set(st.session_state['feature_names']) - set(input_processed.columns)
            for col in missing_cols:
                input_processed[col] = 0
            
            # Réorganiser les colonnes
            input_processed = input_processed[st.session_state['feature_names']]
            
            # Standardisation des données
            input_scaled = st.session_state['scaler'].transform(input_processed)
            
            # Prédiction avec le modèle entraîné
            model = st.session_state['trained_model']
            risk_probability = model.predict_proba(input_scaled)[0][1]
            
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
            
            # Afficher des informations supplémentaires selon le modèle
            st.write(f"**Modèle utilisé :** {st.session_state['model_type']}")
            
            # Graphique de la probabilité
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Probabilité de défaut'], [risk_probability], color='salmon', alpha=0.7)
            ax.barh(['Probabilité de remboursement'], [1-risk_probability], color='lightgreen', alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probabilité')
            ax.set_title('Distribution des Probabilités de Prédiction')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            st.info("Assurez-vous que le modèle a été correctement entraîné et que toutes les colonnes nécessaires sont présentes.")

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
    
    # Vérifier si un modèle est entraîné
    if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
        st.warning("Veuillez d'abord entraîner un modèle pour voir l'importance des caractéristiques.")
    else:
        model = st.session_state['trained_model']
        model_type = st.session_state['model_type']
        
        # Préparer les noms de caractéristiques
        feature_names = st.session_state['feature_names']
        
        # Obtenir l'importance des caractéristiques selon le modèle
        if model_type == "Forêt Aléatoire" or model_type == "Arbre de Décision":
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                
                # Créer un DataFrame pour l'importance
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True)
                
                # Tracer le graphique
                fig, ax = plt.subplots(figsize=(10, 8))
                y_pos = np.arange(len(feature_importance_df))
                ax.barh(y_pos, feature_importance_df['importance'], color='skyblue')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_importance_df['feature'])
                ax.set_xlabel('Importance')
                ax.set_title(f'Importance des Caractéristiques ({model_type})')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Afficher le tableau des importances
                st.write("**Valeurs d'importance détaillées:**")
                st.dataframe(feature_importance_df.sort_values('importance', ascending=False))
            else:
                st.info("L'importance des caractéristiques n'est pas disponible pour ce modèle.")
        else:
            st.info("L'importance des caractéristiques native n'est disponible que pour les modèles Forêt Aléatoire et Arbre de Décision.")
            
        # Pour la régression logistique, on peut afficher les coefficients
        if model_type == "Régression Logistique":
            if hasattr(model, 'coef_'):
                coefficients = model.coef_[0]
                
                # Créer un DataFrame pour les coefficients
                coef_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefficients
                }).sort_values('coefficient', ascending=True)
                
                # Tracer le graphique
                fig, ax = plt.subplots(figsize=(10, 8))
                y_pos = np.arange(len(coef_df))
                colors = ['red' if x < 0 else 'green' for x in coef_df['coefficient']]
                ax.barh(y_pos, coef_df['coefficient'], color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(coef_df['feature'])
                ax.set_xlabel('Coefficient')
                ax.set_title('Coefficients de la Régression Logistique')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Afficher le tableau des coefficients
                st.write("**Coefficients détaillés:**")
                st.dataframe(coef_df.sort_values('coefficient', ascending=False))

# Information sur les modèles
st.sidebar.markdown("---")
st.sidebar.header("À propos des modèles")
st.sidebar.markdown("""
**Forêt Aléatoire**: Ensemble d'arbres de décision, robuste et précis  
**Régression Logistique**: Modèle linéaire, facile à interpréter  
**Arbre de Décision**: Modèle unique, très interprétable
""")
