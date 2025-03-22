!pip install joblib
!pip install -r requirements.txt
import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import spacy
import pdfplumber
import docx
from io import StringIO
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

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
    match = re.search(r'(\d+)\s*(ans|years)', cv_text)
    years_of_experience = int(match.group(1)) if match else 0

    junior_keywords = ["débutant", "junior", "stage", "apprentissage", "assistant"]
    senior_keywords = ["confirmé", "expérience", "middle", "3 ans", "4 ans", "5 ans", "6 ans", "7 ans"]
    expert_keywords = ["lead", "manager", "expert", "architecte", "10 ans", "15 ans", "principal"]

    if any(word in cv_text for word in expert_keywords) or years_of_experience >= 8:
        return "Expert"
    elif any(word in cv_text for word in senior_keywords) or 3 <= years_of_experience < 8:
        return "Senior"
    else:
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
    model_category = joblib.load("/content/drive/MyDrive/model_category.pkl")
    vectorizer = joblib.load("/content/drive/MyDrive/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("/content/drive/MyDrive/label_encoder.pkl")

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
        st.subheader(f"📊 Niveau d'expérience : **{experience}**")

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
