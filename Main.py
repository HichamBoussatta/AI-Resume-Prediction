import spacy
import os

# Vérifier si le modèle est installé, sinon le télécharger
if not os.path.exists("fr_core_news_sm"):
    os.system("python -m spacy download fr_core_news_sm")

# Chargez le modèle spaCy en français
nlp = spacy.load("fr_core_news_sm")
