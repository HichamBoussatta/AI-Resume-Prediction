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
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
# Importations nécessaires
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import shutil
import gdown

# --- Fonction d'authentification ---
def authenticate_user():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Nom d'utilisateur", "")
    password = st.sidebar.text_input("Mot de passe", type="password")

    # Ajouter un utilisateur et un mot de passe pour la démonstration (à sécuriser davantage dans une vraie application)
    correct_username = "admin"
    correct_password = "123"

    error_message_shown = False  # Flag pour vérifier si l'erreur doit être affichée

    if username and password:  # Vérifier si les deux champs sont remplis
        if username != correct_username or password != correct_password:
            error_message_shown = True  # Marquer l'erreur pour qu'elle soit affichée après la tentative

    # Si l'utilisateur essaie de se connecter avec de mauvaises informations
    if error_message_shown:
        st.sidebar.error("Nom d'utilisateur ou mot de passe incorrect.")

    # Vérification des informations de connexion
    if username == correct_username and password == correct_password:
        return True
    else:
        return False


# --- Script 1: Resume Classification Model Training ---
def resume_classification():
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    # Stopwords en français et en anglais
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


    # Télécharger et charger le CSV
    file_url = "https://drive.google.com/uc?id=1-5Hw-uq7-NFJjcU7LuD_pgK42yJc8lxR"
    df = pd.read_csv(gdown.download(file_url, quiet=True), on_bad_lines='skip')

    # Afficher le contenu du DataFrame
    #print(df.head())
    #df = pd.read_csv(r'C:\Users\hicha\Desktop\AI-Resume-Classification\categorized_cvs.csv')


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

    # Affichage de la distribution des classes avant l'équilibrage
    class_distribution_before = df['category'].value_counts().reset_index()
    class_distribution_before.columns = ['Category', 'Count']
    class_distribution_before['Status'] = 'Avant équilibrage'

    # Équilibrage des classes sous-représentées avec RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Affichage de la distribution des classes après l'équilibrage
    class_distribution_after = pd.Series(y_resampled).value_counts().reset_index()
    class_distribution_after.columns = ['Category', 'Count']
    class_distribution_after['Status'] = 'Après équilibrage'

    # Fusionner les deux distributions pour comparaison
    class_distribution_comparison = pd.concat([class_distribution_before, class_distribution_after], axis=0)

    # Interface Streamlit
    st.title("Entraînement des modèles de classification")

    # Affichage du tableau de comparaison des classes avant et après l'équilibrage
    #st.subheader("Comparaison de la distribution des classes avant et après équilibrage")
    #st.dataframe(class_distribution_comparison)

    # Séparation des données en train/test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Fonction d'évaluation des modèles
    def evaluate_model(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        precision = report["macro avg"]["precision"]
        recall = report["macro avg"]["recall"]
        f1 = report["macro avg"]["f1-score"]
        
        return accuracy, precision, recall, f1, report

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

    # Paramètres pour KNeighbors
    knn_n_neighbors = st.sidebar.selectbox("Nombre de voisins (KNeighbors)", [3, 5, 7, 9])

    # Paramètres pour GradientBoosting
    gb_n_estimators = st.sidebar.selectbox("Nombre d'estimateurs (Gradient Boosting)", [100, 200, 300])
    gb_learning_rate = st.sidebar.number_input("Taux d'apprentissage (Gradient Boosting)", min_value=0.01, max_value=1.0, value=0.1)

    # Paramètres pour Naive Bayes
    nb_alpha = st.sidebar.number_input("Alpha (Naive Bayes)", min_value=0.1, max_value=2.0, value=1.0)

    # Variables pour stocker les résultats
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    best_model = None
    best_model_name = ""
    model_trained = False  # Flag pour suivre si les modèles ont été entraînés

    # Définir un chemin local pour sauvegarder les modèles
    models_dir = r'C:\Users\hicha\Desktop\AI-Resume-Classification\Models' # Le dossier "Models" dans le répertoire actuel

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
            acc_logreg, precision_logreg, recall_logreg, f1_logreg, report_logreg = evaluate_model("Logistic Regression", logreg_model, X_test, y_test)
            results['Model'].append('Logistic Regression')
            results['Accuracy'].append(acc_logreg)
            results['Precision'].append(precision_logreg)
            results['Recall'].append(recall_logreg)
            results['F1-Score'].append(f1_logreg)

            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
            rf_model.fit(X_train, y_train)
            acc_rf, precision_rf, recall_rf, f1_rf, report_rf = evaluate_model("Random Forest", rf_model, X_test, y_test)
            results['Model'].append('Random Forest')
            results['Accuracy'].append(acc_rf)
            results['Precision'].append(precision_rf)
            results['Recall'].append(recall_rf)
            results['F1-Score'].append(f1_rf)

            # SVM
            svm_model = SVC(kernel=svm_kernel, class_weight=svm_class_weight)
            svm_model.fit(X_train, y_train)
            acc_svm, precision_svm, recall_svm, f1_svm, report_svm = evaluate_model("SVM", svm_model, X_test, y_test)
            results['Model'].append('SVM')
            results['Accuracy'].append(acc_svm)
            results['Precision'].append(precision_svm)
            results['Recall'].append(recall_svm)
            results['F1-Score'].append(f1_svm)

            # KNeighbors
            knn_model = KNeighborsClassifier(n_neighbors=knn_n_neighbors)
            knn_model.fit(X_train, y_train)
            acc_knn, precision_knn, recall_knn, f1_knn, report_knn = evaluate_model("KNeighbors", knn_model, X_test, y_test)
            results['Model'].append('KNeighbors')
            results['Accuracy'].append(acc_knn)
            results['Precision'].append(precision_knn)
            results['Recall'].append(recall_knn)
            results['F1-Score'].append(f1_knn)

            # Gradient Boosting
            gb_model = GradientBoostingClassifier(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate)
            gb_model.fit(X_train, y_train)
            acc_gb, precision_gb, recall_gb, f1_gb, report_gb = evaluate_model("Gradient Boosting", gb_model, X_test, y_test)
            results['Model'].append('Gradient Boosting')
            results['Accuracy'].append(acc_gb)
            results['Precision'].append(precision_gb)
            results['Recall'].append(recall_gb)
            results['F1-Score'].append(f1_gb)

            # Naive Bayes
            nb_model = MultinomialNB(alpha=nb_alpha)
            nb_model.fit(X_train, y_train)
            acc_nb, precision_nb, recall_nb, f1_nb, report_nb = evaluate_model("Naive Bayes", nb_model, X_test, y_test)
            results['Model'].append('Naive Bayes')
            results['Accuracy'].append(acc_nb)
            results['Precision'].append(precision_nb)
            results['Recall'].append(recall_nb)
            results['F1-Score'].append(f1_nb)

            # Création du DataFrame des résultats
            results_df = pd.DataFrame(results)

            # Affichage des résultats sous forme de tableau
            st.subheader("Tableau des Métriques des Modèles")
            st.dataframe(results_df)

            # Affichage des graphiques séparés pour chaque métrique

            # Graphique pour Accuracy
            #st.subheader("Précision (Accuracy)")
            #fig, ax = plt.subplots(figsize=(10, 5))
            #sns.barplot(x='Model', y='Accuracy', data=results_df, palette="viridis", ax=ax)
            #ax.set_title('Précision (Accuracy)')
            #st.pyplot(fig)

            # Graphique pour Precision
            #st.subheader("Précision (Precision)")
            #fig, ax = plt.subplots(figsize=(10, 5))
            #sns.barplot(x='Model', y='Precision', data=results_df, palette="viridis", ax=ax)
            #ax.set_title('Précision (Precision)')
            #st.pyplot(fig)

            # Graphique pour Recall
            #st.subheader("Rappel (Recall)")
            #fig, ax = plt.subplots(figsize=(10, 5))
            #sns.barplot(x='Model', y='Recall', data=results_df, palette="viridis", ax=ax)
            #ax.set_title('Rappel (Recall)')
            #st.pyplot(fig)

            # Graphique pour F1-Score
            #st.subheader("F1-Score")
            #fig, ax = plt.subplots(figsize=(10, 5))
            #sns.barplot(x='Model', y='F1-Score', data=results_df, palette="viridis", ax=ax)
            #ax.set_title('F1-Score')
            #st.pyplot(fig)

            # Déterminer le meilleur modèle
            best_model_info = max([(logreg_model, "Logistic Regression", acc_logreg),
                                  (rf_model, "Random Forest", acc_rf),
                                  (svm_model, "SVM", acc_svm),
                                  (knn_model, "KNeighbors", acc_knn),
                                  (gb_model, "Gradient Boosting", acc_gb),
                                  (nb_model, "Naive Bayes", acc_nb)], key=lambda x: x[2])

            best_model_name = best_model_info[1]
            best_model = best_model_info[0]
            best_model_acc = best_model_info[2]

            st.write(f"**Meilleur modèle : {best_model_name} avec une précision de {best_model_acc:.4f}**")

            # Sauvegarde du meilleur modèle
            model_path = r'C:\Users\hicha\Desktop\AI-Resume-Classification\Models\best_model.pkl'
            tfidf_path = r'C:\Users\hicha\Desktop\AI-Resume-Classification\Models\tfidf_vectorizer.pkl'
            label_encoder_path = r'C:\Users\hicha\Desktop\AI-Resume-Classification\Models\label_encoder.pkl'


            try:
                st.write(f"Début de la sauvegarde du modèle à {model_path}")
                joblib.dump(best_model, model_path)
                joblib.dump(tfidf_vectorizer, tfidf_path)
                joblib.dump(label_encoder, label_encoder_path)
                st.success(f"✅ Modèle, TfidfVectorizer et LabelEncoder sauvegardés avec succès.")
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
    # Télécharger le modèle si nécessaire
    #os.system("python -m spacy download fr_core_news_sm")
    # Charger le modèle de langage français de SpaCy
    #nlp = spacy.load("fr_core_news_sm")

    def extract_text(uploaded_file):
        # Obtenir le nom du fichier pour vérifier son extension
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file_name.endswith(".docx"):
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

    # 📌 Fonction pour extraire les technologies du CV sans doublons et leur fréquence
    def extract_technologies_with_count(text):
        # Liste complète des compétences à extraire
        tech_keywords = [
            # Data Engineer
            "Python", "SQL", "Spark", "AWS", "Kafka", "Airflow", "Snowflake", "Redshift", "Databricks", "Docker", "Kubernetes", "Jenkins", "ETL", "Pipeline",

            # Data Scientist
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Transformers", "BERT", "LSTM", "GANs", "Reinforcement Learning",

            # Data Analyst
            "Tableau", "Power BI", "SQL", "Excel", "Looker", "Google Data Studio", "DAX", "Pandas", "Matplotlib", "Reporting", "ETL", "Dashboard", "Visualization",

            # DevOps
            "Docker", "Kubernetes", "Terraform", "CI/CD", "Ansible", "Jenkins", "Git", "Helm", "Prometheus", "Grafana", "ArgoCD", "Istio", "OpenShift", "Infrastructure as Code",

            # Backend Developer
            "Java", "Spring Boot", "Microservices", "Hibernate", "REST API", "JPA", "SQL", "NoSQL", "RabbitMQ", "Kafka", "GraphQL", "WebFlux", "Docker", "Kubernetes",

            # Frontend Developer
            "React", "Angular", "Vue.js", "JavaScript", "TypeScript", "HTML", "CSS", "Redux", "Next.js", "Nuxt.js", "Tailwind", "Material-UI", "Cypress", "Jest", "Storybook", "Webpack",

            # Mobile Developer
            "Swift", "Kotlin", "Flutter", "React Native", "Android", "iOS", "Jetpack Compose", "SwiftUI", "Objective-C", "Dart", "Mobile UI/UX", "GraphQL", "Firebase",

            # Big Data Engineer
            "Hadoop", "Hive", "Pig", "Scala", "Spark", "Presto", "Flink", "HBase", "Cassandra", "ElasticSearch", "MapReduce", "Delta Lake", "Kudu", "YARN", "Zookeeper",

            # Cybersecurity Analyst
            "Cybersecurity", "Penetration Testing", "SIEM", "IDS/IPS", "Firewall", "Ethical Hacking", "Kali Linux", "Metasploit", "OWASP", "Burp Suite", "Nmap", "Security Compliance",

            # ETL Developer
            "ETL", "Talend", "SSIS", "Informatica", "DataStage", "Azure Data Factory", "Apache Nifi", "Pentaho", "Snowflake", "SQL", "Data Warehouse", "OLAP",

            # Database Administrator
            "Oracle", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "MariaDB", "CockroachDB", "Elasticsearch", "TimescaleDB", "SQL", "NoSQL", "Database Replication", "Indexing",

            # CRM Consultant
            "Salesforce", "SAP", "ERP", "CRM", "Dynamics 365", "Zoho CRM", "HubSpot", "Workday", "NetSuite", "ServiceNow", "Business Process Automation", "RPA",

            # Business Intelligence Analyst
            "Excel", "VBA", "R", "Power BI", "Tableau", "SAS", "QlikView", "SQL", "Alteryx", "ETL", "Reporting", "KPI", "Business Intelligence", "Dashboarding",

            # Cloud Engineer
            "GCP", "AWS", "Azure", "Terraform", "Ansible", "CloudFormation", "Kubernetes", "Lambda", "Serverless", "DevOps", "Cloud Security", "IAM", "Networking",

            # AI Researcher
            "Computer Vision", "OpenCV", "YOLO", "Deep Learning", "TensorFlow", "PyTorch", "GANs", "Image Segmentation", "Object Detection", "Image Processing",

            # Blockchain Developer
            "Blockchain", "Ethereum", "Smart Contracts", "Solidity", "Hyperledger", "Polkadot", "Binance Smart Chain", "DeFi", "dApps", "NFTs", "Consensus Mechanisms",

            # IoT Engineer
            "IoT", "MQTT", "Edge Computing", "Raspberry Pi", "Arduino", "LoRaWAN", "Zigbee", "Smart Devices", "Industrial IoT", "IoT Security",

            # Network Engineer
            "Networking", "TCP/IP", "BGP", "OSPF", "SDN", "Cisco", "Juniper", "Wireshark", "Routing", "Switching", "Network Security", "VLAN", "MPLS",

            # IT Project Manager
            "Project Management", "Agile", "Scrum", "PMP", "Kanban", "Jira", "Confluence", "SAFe", "Prince2", "Risk Management", "Stakeholder Management", "Product Ownership"
        ]
        
        # Créer une expression régulière pour chaque technologie (en échappant les espaces et les parenthèses)
        tech_regex = [re.escape(tech.lower()) for tech in tech_keywords]
        combined_regex = r'\b(' + r'|'.join(tech_regex) + r')\b'

        # Trouver toutes les correspondances dans le texte (insensible à la casse)
        found_technologies = re.findall(combined_regex, text.lower())

        # Compter la fréquence des technologies, sans doublons
        tech_counts = Counter(found_technologies)

        # Retourner la liste des technologies avec leur fréquence sous forme de dictionnaire
        return dict(tech_counts)

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
        
    # 📌 Fonction pour comparer les technologies avec un répertoire de CVs
    def compare_technologies_with_cvs(job_description, cvs_directory=r'C:\Users\hicha\Desktop\AI-Resume-Classification\BaseCVs'):
        # Extraire les technologies de l'offre d'emploi
        job_tech_counts = extract_technologies_with_count(job_description)

        # Initialisation des résultats
        matching_cvs = []

        # Parcours des fichiers dans le répertoire
        for cv_file in os.listdir(cvs_directory):
            cv_path = os.path.join(cvs_directory, cv_file)
            if cv_path.endswith(('.pdf', '.docx')):  # Accepter seulement les fichiers PDF et DOCX
                with open(cv_path, 'rb') as file:
                    cv_text = extract_text(file)
                    tech_counts = extract_technologies_with_count(cv_text)

                    # Comparer les technologies du CV avec celles de l'offre d'emploi
                    common_tech = set(tech_counts.keys()).intersection(set(job_tech_counts.keys()))
                    if common_tech:
                        # Calculer la similarité entre le CV et l'offre d'emploi
                        similarity_score = compare_texts(cv_text, job_description)
                        matching_cvs.append({
                            'cv_file': cv_file,
                            'common_tech': common_tech,
                            'tech_counts': tech_counts,
                            'job_tech_counts': job_tech_counts,
                            'similarity_score': similarity_score
                        })
        
        return matching_cvs
    
    # 📌 Fonction pour enregistrer les résultats dans un fichier
    def export_matching_cvs(matching_cvs, cvs_directory=r'C:\Users\hicha\Desktop\AI-Resume-Classification\BaseCVs', output_directory=r'C:\Users\hicha\Desktop\AI-Resume-Classification\Output'):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Créer un ensemble pour suivre les fichiers déjà exportés
        exported_files = set()

        # Exporter les CVs complets dans le répertoire de sortie
        for match in matching_cvs:
            cv_file = match['cv_file']
            cv_path = os.path.join(cvs_directory, cv_file)

            # Vérifier si le fichier a déjà été exporté
            if cv_file not in exported_files:
                # Copier le CV original dans le répertoire de sortie
                shutil.copy(cv_path, output_directory)
                
                # Ajouter le fichier au set des fichiers exportés
                exported_files.add(cv_file)
                
                # Ajouter un fichier texte avec les technologies communes
                output_file_path = os.path.join(output_directory, f"match_{cv_file}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Fichier CV: {cv_file}\n")
                    f.write(f"Technologies communes avec l'offre d'emploi: {', '.join(match['common_tech'])}\n")



    # 📌 Fonction pour comparer les technologies détectées avec l'offre d'emploi
    def compare_technologies_with_job(tech_counts, job_description):
        # Extraire les technologies de l'offre d'emploi
        job_tech_counts = extract_technologies_with_count(job_description)

        # Calculer le nombre de technologies communes entre le CV et l'offre d'emploi
        common_tech = set(tech_counts.keys()).intersection(set(job_tech_counts.keys()))
        common_tech_count = sum([min(tech_counts[tech], job_tech_counts[tech]) for tech in common_tech])

        return common_tech_count

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
        # Charger les modèles et autres objets depuis le répertoire local
        model_category = joblib.load(r"C:\Users\hicha\Desktop\AI-Resume-Classification\Models\best_model.pkl")
        vectorizer = joblib.load(r"C:\Users\hicha\Desktop\AI-Resume-Classification\Models\tfidf_vectorizer.pkl")
        label_encoder = joblib.load(r"C:\Users\hicha\Desktop\AI-Resume-Classification\Models\label_encoder.pkl")

        # Traitement du texte du CV
        cleaned_text = clean_text(cv_text)
        features = vectorizer.transform([cleaned_text])

        # Prédiction de la catégorie et de l'expérience
        predicted_category_encoded = model_category.predict(features)[0]
        predicted_category = label_encoder.inverse_transform([predicted_category_encoded])[0]
        predicted_experience = detect_experience(cv_text)

        return predicted_category, predicted_experience

    # 📌 Interface Streamlit
    st.title("📄🤖 CV Analysis & Job Match")
    st.subheader("Chargez un CV et obtenez la prédiction du métier et de l'expérience.")

    uploaded_file = st.file_uploader("📤 Importer un fichier (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    # 🔥 L'utilisateur peut définir son propre profil idéal
    st.subheader("🔧 Définissez votre profil idéal")
    available_skills = ["Python","SQL","Sql","Spark","AWS","Kafka","Airflow","Snowflake","Redshift","Databricks","Docker","Kubernetes","Jenkins","ETL","Pipeline","Machine Learning",
            "Deep Learning","NLP","Computer Vision","TensorFlow","PyTorch","Keras","Scikit-learn","Transformers","BERT","LSTM","GANs","Reinforcement Learning","Tableau",
            "Power BI","Excel","Looker","Google Data Studio","DAX","Pandas","Matplotlib","Reporting","Dashboard","Visualization","Terraform","CI/CD","Ansible","Git","Helm","Prometheus","Grafana","ArgoCD","Istio",
            "OpenShift","Infrastructure as Code","Java","Spring Boot","Microservices","Hibernate","REST API","JPA","NoSQL","RabbitMQ","GraphQL","WebFlux","React","Angular","Vue.js","JavaScript","TypeScript","HTML","CSS","Redux","Next.js","Nuxt.js","Tailwind",
            "Material-UI","Cypress","Jest","Storybook","Webpack","Swift","Kotlin","Flutter","React Native","Android","iOS","Jetpack Compose","SwiftUI","Objective-C","Dart",
            "Mobile UI/UX","Firebase","Hadoop","Hive","Pig","Scala","Presto","Flink","HBase","Cassandra","ElasticSearch","MapReduce","Delta Lake","Kudu","YARN","Zookeeper","Cybersecurity","Penetration Testing","SIEM","IDS/IPS","Firewall","Ethical Hacking","Kali Linux","Metasploit","OWASP",
            "Burp Suite","Nmap","Security Compliance","Talend","SSIS","Informatica","DataStage","Azure Data Factory","Apache Nifi","Pentaho",
            "Data Warehouse","OLAP","Oracle","MySQL","PostgreSQL","MongoDB","Redis","MariaDB","CockroachDB","Elasticsearch","TimescaleDB",
            "Database Replication","Indexing","Salesforce","SAP","ERP","CRM","Dynamics 365","Zoho CRM","HubSpot","Workday","NetSuite","ServiceNow",
            "Business Process Automation","RPA","VBA","R","SAS","QlikView","Alteryx","KPI","Business Intelligence","Dashboarding","GCP","Azure","CloudFormation","Lambda","Serverless","DevOps","Cloud Security","IAM","Networking","OpenCV","YOLO",
            "Image Segmentation","Object Detection","Image Processing","Blockchain","Ethereum","Smart Contracts","Solidity","Hyperledger","Polkadot",
            "Binance Smart Chain","DeFi","dApps","NFTs","Consensus Mechanisms","IoT","MQTT","Edge Computing","Raspberry Pi","Arduino","LoRaWAN","Zigbee","Smart Devices",
            "Industrial IoT","IoT Security","TCP/IP","BGP","OSPF","SDN","Cisco","Juniper","Wireshark","Routing","Switching","Network Security","VLAN","MPLS",
            "Project Management","Agile","Scrum","PMP","Kanban","Jira","Confluence","SAFe","Prince2","Risk Management","Stakeholder Management","Product Ownership"]

    selected_skills = st.multiselect("Sélectionnez les compétences clés :", available_skills)
    ideal_profile = {skill: 1 for skill in selected_skills}

    job_url = st.text_input("Entrez l'URL de l'offre d'emploi :", "")

    # Exemple d'URL à afficher sous le champ de saisie (non cliquable)
    st.markdown("""
    *Exemple d'URL de la plateforme Emploi.ma :*  
    https://www.emploi.ma/offre-emploi-maroc/senior-data-engineers-hf-casablanca-rabat-8802045
    """)

    if uploaded_file:
        with st.spinner("🔍 Analyse en cours..."):
            cv_text = extract_text(uploaded_file)
            cleaned_cv_text = clean_text(cv_text)

            st.subheader("🔍 Contenu du CV :")
            st.text_area("Texte extrait :", cv_text, height=200)

            category, experience = predict_cv(cv_text)
            st.subheader(f"🎯 Métier : **{category}**")

            # Extraction des technologies et comptage
            tech_counts = extract_technologies_with_count(cv_text)

            if tech_counts:
                st.subheader("🛠️ Technologies détectées et leurs fréquences :")
                
                # Trier les technologies par fréquence (du plus élevé au plus bas)
                sorted_tech_counts = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Prendre uniquement les 10 premiers éléments
                top_10_tech = sorted_tech_counts[:10]

                # Appliquer la transformation pour mettre la première lettre en majuscule
                top_10_tech = [(tech.capitalize(), count) for tech, count in top_10_tech]
                
                # Affichage sous forme de tableau
                tech_data = [{"Technologie": tech, "Fréquence": count} for tech, count in top_10_tech]
                st.table(tech_data)
                
                # Affichage avec un graphique à barres
                tech_names = [tech for tech, count in top_10_tech]
                tech_frequencies = [count for tech, count in top_10_tech]
                
                
                # Créer un graphique à barres
                #plt.figure(figsize=(10, 6))
                #sns.barplot(x=tech_names, y=tech_frequencies, palette="viridis")
                #plt.title("Top 10 des technologies détectées")
                #plt.xlabel("Technologies")
                #plt.ylabel("Fréquence")
                #plt.xticks(rotation=45, ha="right")
                
                #st.pyplot(plt)
                
            else:
                st.subheader("🛠️ Aucune technologie détectée")

            # 🔥 Ajout du graphique radar
            st.subheader("📊 Visualisation des compétences")

            # Prendre uniquement les 10 technologies les plus fréquentes
            top_10_tech = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Appliquer la transformation pour mettre la première lettre en majuscule
            top_10_tech = [(tech.capitalize(), count) for tech, count in top_10_tech]
            
            user_skills = {tech: count for tech, count in top_10_tech}  # Utiliser seulement les 10 meilleures compétences

            # Visualisation avec le graphique radar
            plot_radar_chart(user_skills, ideal_profile)

            # Extraction des technologies et comptage
            tech_counts = extract_technologies_with_count(cv_text)

            # Vérification si l'URL de l'offre d'emploi est renseignée
            if job_url:
                # Scraper l'offre d'emploi
                job_description = scrape_job_description(job_url)

                if job_description:
                    st.subheader("🔍 Comparaison des technologies de l'offre avec les CVs :")
                    
                    # Comparer les technologies du CV avec celles des CVs dans le répertoire
                    matching_cvs = compare_technologies_with_cvs(job_description)

                    if matching_cvs:
                        # Afficher un tableau avec les résultats
                        st.write("Technologies communes trouvées dans les CVs :")
                        data = []
                        for match in matching_cvs:
                            data.append({
                                "Nom du CV": match["cv_file"],
                                "Technologies communes": ", ".join(match["common_tech"]),
                                "Score de Similarité": round(match["similarity_score"], 2)
                            })
                        # Afficher le tableau avec les informations
                        #st.write(pd.DataFrame(data))
                        # Afficher un tableau avec les résultats
                        st.write(pd.DataFrame(data).sort_values(by="Score de Similarité", ascending=False))


                        # Exporter les CVs originaux et les informations dans un répertoire
                        export_matching_cvs(matching_cvs)

                        # Afficher un message indiquant que les résultats ont été exportés
                        st.success(f"Les CVs originaux ont été exportés dans le dossier")
                    else:
                        st.write("Aucune technologie commune trouvée dans les CVs.")
                else:
                    st.error("Erreur lors du scraping de l'offre d'emploi.")
            else:
                st.write("Aucune URL d'offre d'emploi fournie. Pas de comparaison effectuée.")


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

# --- Fonction principale avec les onglets stylisés ---
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

    # Sécuriser l'option "Resume Classification"
    if option == "Resume Classification":
        # Demander l'authentification avant d'accéder à cette option
        if authenticate_user():
            resume_classification()  # Si authentification réussie, afficher le contenu
        else:
            st.title("📂✨ Smart Resume Classification")
            st.write("Accès non autorisé. Veuillez entrer vos informations de connexion.")
    
    elif option == "CV Analysis and Job Match":
        automated_cv_analysis()  # Cette option reste publique

if __name__ == "__main__":
    main()
