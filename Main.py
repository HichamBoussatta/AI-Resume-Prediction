import spacy
import streamlit as st
from spacy import displacy

# Chargez le modèle spaCy en français
nlp = spacy.load("fr_core_news_sm")

# Titre de l'application Streamlit
st.title("Visualisation spaCy avec Streamlit")

# Ajoutez un champ de texte où l'utilisateur peut saisir une phrase
text = st.text_area("Entrez du texte :", "Jean Dupont travaille chez VISEO et il vit à Rabat.")

# Traitez le texte avec spaCy
doc = nlp(text)

# Affichez les entités nommées
if st.button('Afficher les entités nommées'):
    st.subheader("Entités nommées :")
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    st.write(entities)

# Visualisation des dépendances avec displacy
if st.button('Afficher la visualisation des dépendances'):
    st.subheader("Visualisation des dépendances grammaticales :")
    displacy.render(doc, style="dep", page=True)
