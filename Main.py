import spacy
import os

# Vérifier si le modèle est installé, sinon l'installer
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

print("Modèle spaCy chargé avec succès !")
