import spacy
import os

# Vérifier si le modèle est déjà installé, sinon le télécharger
if not os.path.exists(spacy.util.get_data_path() + "/fr_core_news_sm"):
    os.system("python -m spacy download fr_core_news_sm")

# Charger le modèle spaCy
nlp = spacy.load("fr_core_news_sm")
