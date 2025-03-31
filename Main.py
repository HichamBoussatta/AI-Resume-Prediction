import spacy
import os

# Tenter de charger le modèle spaCy, s'il échoue, télécharger le modèle
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("Le modèle fr_core_news_sm n'est pas trouvé, téléchargement...")
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")
