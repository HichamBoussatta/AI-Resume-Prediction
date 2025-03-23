import streamlit as st
import joblib
import os
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
import spacy
import pdfplumber
import docx
from io import StringIO
from wordcloud import WordCloud
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from datetime import datetime

# --- Script 1: Resume Classification Model Training ---
def resume_classification():

    # Télécharger les stopwords français
    nltk.download("stopwords")
    french_stop_words = set(stopwords.words("french"))
    english_stop_words = ENGLISH_STOP_WORDS  # Stopwords en anglais
    all_stop_words = french_stop_words.union(english_stop_words)

    # Fonction avancée de nettoyage du texte
    def clean_text(text):
        if pd.isna(text):  # Gérer les valeurs NaN
            return ""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Supprimer les caractères spéciaux et chiffres
        text = re.sub(r"\s+", " ", text).strip()  # Supprimer les espaces supplémentaires
        text = " ".join([word for word in text.split() if word not in all_stop_words])  # Supprimer les stopwords
        return text

    # Charger les données
    df = pd.read_csv('/content/drive/MyDrive/categorized_cvs.csv')

    # Suppression des lignes avec des valeurs manquantes
    df.dropna(subset=['category', 'text'], inplace=True)

    # Nettoyage avancé du texte
    df["clean_text"] = df["text"].apply(clean_text)

    # Encodage des catégories
    label_encoder = LabelEncoder()
    df["category_encoded"] = label_encoder.fit_transform(df["category"])

    # Transformation du texte en vecteurs TF-IDF (bigrammes + 5000 features)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf_vectorizer.fit_transform(df["clean_text"])
    y = df["category_encoded"]

    # Équilibrage des classes sous-représentées avec RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Séparation des données en train/test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Fonction d'évaluation des modèles
    def evaluate_model(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    # Interface Streamlit
    st.title("Entraînement des modèles de classification")

    # Section pour les hyperparamètres
    st.sidebar.header("Paramètres des modèles")

    # Paramètres pour Logistic Regression
    logreg_max_iter = st.sidebar.number_input("Max Iterations (Logistic Regression)", min_value=100, max_value=5000, value=2000)
    logreg_class_weight = st.sidebar.selectbox("Poids des classes (Logistic Regression)", ["balanced", "None"])

    # Paramètres pour Random Forest
    rf_n_estimators = st.sidebar.selectbox("Nombre d'estimateurs (RandomForest)", [100, 200, 300])
    rf_max_depth = st.sidebar.selectbox("Profondeur maximale (RandomForest)", [10, 20, None])

    # Paramètres pour SVM
    svm_kernel = st.sidebar.selectbox("Noyau (SVM)", ["linear", "rbf"])
    svm_class_weight = st.sidebar.selectbox("Poids des classes (SVM)", ["balanced", "None"])

    # Variables pour stocker les résultats
    acc_logreg, acc_rf, acc_svm = 0, 0, 0
    best_model = None
    best_model_name = ""
    model_trained = False  # Flag pour suivre si les modèles ont été entraînés

    # Définir un chemin local pour sauvegarder les modèles
    models_dir = './Models'  # Le dossier "Models" dans le répertoire actuel


    # Bouton pour entraîner les modèles
    if st.sidebar.button("Entraîner les modèles"):
          # Vérifier le répertoire Models et le créer s'il n'existe pas
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            st.write(f"Répertoire '{models_dir}' créé.")
        else:
            st.write(f"Répertoire '{models_dir}' déjà existant.")
        with st.spinner("Les modèles sont en cours d'entraînement..."):
            # Régression logistique
            logreg_model = LogisticRegression(max_iter=logreg_max_iter, class_weight=logreg_class_weight)
            logreg_model.fit(X_train, y_train)
            acc_logreg, report_logreg = evaluate_model("Logistic Regression", logreg_model, X_test, y_test)

            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
            rf_model.fit(X_train, y_train)
            acc_rf, report_rf = evaluate_model("Random Forest", rf_model, X_test, y_test)

            # SVM
            svm_model = SVC(kernel=svm_kernel, class_weight=svm_class_weight)
            svm_model.fit(X_train, y_train)
            acc_svm, report_svm = evaluate_model("SVM", svm_model, X_test, y_test)

            # Affichage des résultats
            results_df = pd.DataFrame({
                'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
                'Accuracy': [acc_logreg, acc_rf, acc_svm]
            })
            st.subheader("Tableau des Précisions des Modèles")
            st.dataframe(results_df)

            # Affichage du graphique avec Seaborn
            st.subheader("Graphique des Précisions des Modèles")
            fig, ax = plt.subplots()
            sns.barplot(x='Model', y='Accuracy', data=results_df, palette="viridis", ax=ax)
            ax.set_title("Comparaison des Précisions des Modèles")
            st.pyplot(fig)

            # Déterminer le meilleur modèle
            best_model_info = max([(logreg_model, "Logistic Regression", acc_logreg),
                                  (rf_model, "Random Forest", acc_rf),
                                  (svm_model, "SVM", acc_svm)], key=lambda x: x[2])

            best_model_name = best_model_info[1]
            best_model = best_model_info[0]
            best_model_acc = best_model_info[2]

            st.write(f"**Meilleur modèle : {best_model_name} avec une précision de {best_model_acc:.4f}**")

            # Sauvegarde du meilleur modèle
            model = './Models/...'
            model_path = './Models/best_model.pkl'
            tfidf_path = './Models/tfidf_vectorizer.pkl'
            label_encoder_path = './Models/label_encoder.pkl'

            try:
                st.write(f"Début de la sauvegarde du modèle à {model_path}")
                joblib.dump(best_model, model_path)
                joblib.dump(tfidf_vectorizer, tfidf_path)
                joblib.dump(label_encoder, label_encoder_path)
                st.success(f"✅ Modèle, TfidfVectorizer et LabelEncoder sauvegardés avec succès. à : {model}")
            except FileNotFoundError as fnf_error:
                st.error(f"❌ Erreur FileNotFoundError : {str(fnf_error)}")
            except PermissionError as perm_error:
                st.error(f"❌ Erreur PermissionError : {str(perm_error)}")
            except Exception as e:
                st.error(f"❌ Erreur inconnue lors de la sauvegarde du modèle : {str(e)}")


# --- Script 2: Automated CV Analysis and Job Match ---
def automated_cv_analysis():

    # Télécharger les stopwords
    nltk.download("stopwords")

    # Charger le modèle de langage français de SpaCy
    nlp = spacy.load("fr_core_news_sm")

    # 📌 Fonction pour extraire le texte d'un fichier PDF, DOCX ou TXT
    def extract_text(uploaded_file):
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return " ".join([para.text for para in doc.paragraphs])
        else:
            return StringIO(uploaded_file.getvalue().decode("utf-8")).read()

    # 📌 Fonction de nettoyage du texte
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"\d+", "", text)  # Supprimer les chiffres
        text = re.sub(r"[^\w\s]", "", text)  # Supprimer la ponctuation
        text = " ".join([word for word in text.split() if word not in stopwords.words("french")])
        return text

    # 📌 Fonction pour extraire les technologies du CV
    def extract_technologies(text):
        tech_keywords = ["python", "sql", "spark", "aws", "machine learning", "deep learning",
                        "tableau", "power bi", "docker", "kubernetes", "terraform", "ci/cd"]
        return [tech for tech in tech_keywords if tech in text.lower()]

    # 📌 Fonction pour détecter le niveau d'expérience

    def detect_experience(cv_text):
        cv_text = cv_text.lower()

        # Détecter les années d'expérience explicites
        match = re.search(r'(\d+)\s*(ans|years)', cv_text)
        years_of_experience = int(match.group(1)) if match else 0

        # Listes de mots-clés pour classifier
        junior_keywords = ["débutant", "junior", "stage", "apprentissage", "assistant"]
        senior_keywords = ["confirmé", "middle", "expérience significative", "3 ans", "4 ans", "5 ans", "6 ans", "7 ans"]
        expert_keywords = ["lead", "manager", "expert", "architecte", "10 ans", "15 ans", "principal"]

        # Déterminer le niveau basé sur les années détectées
        if years_of_experience >= 8:
            return "Expert"
        elif 3 <= years_of_experience < 8:
            return "Senior"
        elif years_of_experience > 0:
            return "Junior"

        # Si aucune année n'est détectée, on vérifie les mots-clés supplémentaires
        if any(word in cv_text for word in expert_keywords):
            return "Expert"
        elif any(word in cv_text for word in senior_keywords):
            return "Senior"
        elif any(word in cv_text for word in junior_keywords):
            return "Junior"

        # Par défaut, classer comme Junior si aucune correspondance n'est trouvée
        return "Junior"

    # 📌 Fonction pour créer un nuage de mots
    def generate_wordcloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # 📌 Fonction pour générer un graphique radar des compétences
    def plot_radar_chart(user_skills, ideal_profile):
        labels = list(set(user_skills.keys()).union(set(ideal_profile.keys())))
        user_values = [user_skills.get(skill, 0) for skill in labels]
        ideal_values = [ideal_profile.get(skill, 0) for skill in labels]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=user_values, theta=labels, fill='toself', name='CV'))
        fig.add_trace(go.Scatterpolar(r=ideal_values, theta=labels, fill='toself', name='Profil Idéal'))
        
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig)

    # 📌 Fonction pour scraper l'offre d'emploi
    def scrape_job_description(url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extraire les sections pertinentes
            description_section = soup.find('div', class_='job-description')
            qualifications_section = soup.find('div', class_='job-qualifications')
            criteria_section = soup.find('ul', class_='arrow-list')
            
            # Concatenation des sections
            description = ""
            if description_section:
                description += description_section.get_text(separator="\n", strip=True) + "\n"
            if qualifications_section:
                description += qualifications_section.get_text(separator="\n", strip=True) + "\n"
            if criteria_section:
                description += criteria_section.get_text(separator="\n", strip=True) + "\n"
            
            return description
        else:
            return ""

    # 📌 Fonction pour obtenir les mots les plus fréquents dans un texte
    def get_top_keywords(text, num_keywords=5):
        words = text.split()
        words = [word for word in words if word not in stopwords.words("french")]  # Supprimer les stopwords
        word_counts = Counter(words)
        return word_counts.most_common(num_keywords)

    # 📌 Fonction pour créer un nuage de mots à partir des mots les plus fréquents
    def generate_top_keywords_wordcloud(text):
        # Extraire les 5 mots les plus fréquents
        top_keywords = get_top_keywords(text, num_keywords=5)
        
        # Créer un dictionnaire avec les mots et leurs fréquences
        word_freq = dict(top_keywords)

        # Générer le nuage de mots avec les fréquences
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

        # Affichage du nuage de mots
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # 📌 Fonction pour comparer les textes
    def compare_texts(text1, text2):
        # Vectorisation des textes
        vectorizer = CountVectorizer().fit_transform([text1, text2])
        cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
        
        return cosine_sim[0][0]

    # 📌 Fonction pour prédire la catégorie et l'expérience en utilisant le modèle préalablement entraîné
    def predict_cv(cv_text):
        model_category = joblib.load("/content/Models/best_model.pkl")
        vectorizer = joblib.load("/content/Models/tfidf_vectorizer.pkl")
        label_encoder = joblib.load("/content/Models/label_encoder.pkl")

        cleaned_text = clean_text(cv_text)
        features = vectorizer.transform([cleaned_text])

        predicted_category_encoded = model_category.predict(features)[0]
        predicted_category = label_encoder.inverse_transform([predicted_category_encoded])[0]
        predicted_experience = detect_experience(cv_text)

        return predicted_category, predicted_experience

    # 📌 Interface Streamlit
    st.title("📄💡 Classification Automatique des CVs")
    st.subheader("Chargez un CV et obtenez la prédiction du métier et de l'expérience.")

    uploaded_file = st.file_uploader("📤 Importer un fichier (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    # 🔥 L'utilisateur peut définir son propre profil idéal
    st.subheader("🔧 Définissez votre profil idéal")
    available_skills = ["python", "sql", "spark", "aws", "machine learning", "deep learning", "tableau", "power bi", "docker", "kubernetes", "terraform", "ci/cd"]
    selected_skills = st.multiselect("Sélectionnez les compétences clés :", available_skills)
    ideal_profile = {skill: 1 for skill in selected_skills}

    job_url = st.text_input("Entrez l'URL de l'offre d'emploi :", "")


    if uploaded_file:
        with st.spinner("🔍 Analyse en cours..."):
            cv_text = extract_text(uploaded_file)
            cleaned_cv_text = clean_text(cv_text)

            st.subheader("🔍 Contenu du CV :")
            st.text_area("Texte extrait :", cv_text, height=200)

            st.subheader("☁️ Nuage de mots :")
            generate_wordcloud(cleaned_cv_text)

            category, experience = predict_cv(cv_text)
            st.subheader(f"🎯 Métier : **{category}**")
            #st.subheader(f"📊 Niveau d'expérience : **{experience}**")

            technologies = extract_technologies(cv_text)
            st.subheader(f"🛠️ Technologies détectées : {', '.join(technologies) if technologies else 'Aucune'}")

            # 🔥 Ajout du graphique radar
            st.subheader("📊 Visualisation des compétences")
            user_skills = {tech: 1 for tech in technologies}
            plot_radar_chart(user_skills, ideal_profile)

            # Scraper l'offre d'emploi
            job_description = scrape_job_description(job_url)
            
            if job_description:
                st.subheader("🔍 Nuage de mots des 5 mots les plus fréquents dans l'offre d'emploi :")
                job_description_cleaned = clean_text(job_description)
                generate_top_keywords_wordcloud(job_description_cleaned)
                
                # Comparer le CV avec l'offre d'emploi
                similarity_score = compare_texts(cv_text, job_description)
                st.subheader("Résultats de la comparaison :")
                st.write(f"Score de similarité entre le CV et l'offre d'emploi : {similarity_score:.4f}")
                
                # Interprétation du score de similarité
                if similarity_score > 0.8:
                    st.success("Le CV est très similaire à l'offre d'emploi.")
                elif similarity_score > 0.5:
                    st.warning("Le CV est modérément similaire à l'offre d'emploi.")
                else:
                    st.error("Le CV est peu similaire à l'offre d'emploi.")
            else:
                st.error("Erreur lors du scraping de l'offre d'emploi.")

# Fonction pour afficher la page de connexion
def login_page():
    st.title("Page de Connexion")

    # Champ pour le nom d'utilisateur
    username = st.text_input("Nom d'utilisateur")

    # Champ pour le mot de passe
    password = st.text_input("Mot de passe", type="password")

    # Vérifier les identifiants
    if st.button("Se connecter"):
        if username == "Hicham" and password == "1992":  # Vous pouvez ajuster ces identifiants
            st.success("Connexion réussie !")
            return True
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")
            return False

    return False


def set_sidebar_style():
    st.markdown("""
    <style>
        /* Style pour la barre latérale */
        .sidebar .sidebar-content {
            background-color: #000000; /* Arrière-plan noir pour la sidebar */
            color: white;
        }

        /* Style du titre dans la sidebar */
        .sidebar .sidebar-content h1, .sidebar .sidebar-content h2 {
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }

        /* Style des radio buttons */
        .stRadio>div {
            background-color: #333333; /* Fond gris foncé pour les boutons */
            color: white; 
            border: 2px solid #ffffff;  /* Bordure blanche */
            border-radius: 8px;
            padding: 12px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }

        /* Style quand le bouton est survolé */
        .stRadio>div:hover {
            background-color: #555555; /* Changer la couleur de fond au survol */
        }

        /* Style du texte dans les boutons radio */
        .stRadio>div>label {
            color: white; /* Texte blanc pour les labels */
        }
    </style>
    """, unsafe_allow_html=True)

# Fonction principale avec les onglets
def main():
    set_sidebar_style()  # Appliquer les styles

    st.sidebar.title("Navigation")

    # Utiliser des boutons radio pour les onglets (afin de mieux les styliser)
    option = st.sidebar.radio(
        "Choisissez un onglet",
        ("Resume Classification", "CV Analysis and Job Match"),
        index=0,  # Index de l'onglet par défaut
        key="sidebar_radio"
    )

    # Si l'option choisie est "Resume Classification"
    if option == "Resume Classification":
        # Afficher la page de connexion
        if login_page():
            resume_classification()  # Afficher le contenu sécurisé
        else:
            st.warning("Vous devez être connecté pour accéder à Resume Classification.")
    
    # Si l'option choisie est "CV Analysis and Job Match"
    elif option == "CV Analysis and Job Match":
        automated_cv_analysis()  # Afficher le contenu public

if __name__ == "__main__":
    main()


