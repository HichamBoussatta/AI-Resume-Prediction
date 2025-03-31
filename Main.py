import spacy
import os
import subprocess

# Tenter de charger le modèle spaCy, sinon le télécharger
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("Le modèle fr_core_news_sm n'est pas trouvé, téléchargement...")
    # Télécharger le modèle avec subprocess
    subprocess.check_call([os.sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
    nlp = spacy.load("fr_core_news_sm")
