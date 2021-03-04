# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:34:15 2021

@author: 33664
"""
import streamlit as st

import tensorflow as tf

import numpy as np
#nltk.download("stopwords") 
#nltk.download('punkt')
from func import macro_soft_f1,macro_f1, preprocess_sentence

from tensorflow.keras.models import load_model
import pickle
import os
# Récupérer le chemin de ce script

@st.cache
def tokenize_text(text):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    tokenized_text=tokenizer.texts_to_sequences([preprocess_sentence(text)])
    prepro_text=tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=17, padding='post')[0]
    prepro_text=prepro_text.reshape(1,17)
    return prepro_text

@st.cache
def preprocess_text(text):
    return preprocess_sentence(text)

 

html_temp = """
<div >
<h1 style="color:rgb(191,0,0);text-align:center;">Classification à partir des données textes </h1>
</div>
"""
html_temp1 = """
<div >
<h2 style="color:black;font-weight: bold;">Word cloud des données</h2>
</div>
"""
html_temp2 = """
<div >
<h2 style="color:black;font-weight: bold;">Désignation de produit</h2>
</div>
"""
html_temp3 = """
<div >
<h2 style="color:black;font-weight: bold;">Prétraitement</h2>
</div>
"""
html_temp4 = """
<div >
<h2 style="color:black;font-weight: bold;">Classification</h2>
</div>
"""


st.markdown(html_temp,unsafe_allow_html=True)
@st.cache 
def predict(text):
    bi_lstm_200=load_model("rnn.h5", custom_objects={'macro_soft_f1': macro_soft_f1, "macro_f1":macro_f1})
    y_pred=bi_lstm_200.predict(text)
    classe=np.argmax(y_pred,axis = 1)
    return classe


st.text("")
st.markdown(html_temp1,unsafe_allow_html=True)
wc = st.beta_container()


st.markdown(html_temp2,unsafe_allow_html=True)
designation = st.beta_container()
st.markdown(html_temp3,unsafe_allow_html=True)
prepro = st.beta_container()
st.markdown(html_temp4,unsafe_allow_html=True)
classif = st.beta_container()


# ------------------------------- BACKEND -------------------------------------
#♣ Word cloud
paths=['avant_prepro.png','apres_prepro.png']
st.text("")
wc.image(paths,  use_column_width=True,caption=["Avant preprocessing", "Après preprocessing"])
st.text("")

# entrer designation
texte=designation.text_input("Entrez la désignation","Collez ou tapez votre texte")
st.text("")

#preprocessing
st.text("")
propro_text=preprocess_text(texte)
prepro.subheader("Après préprocessing nous obtenons :")
prepro.markdown(propro_text)
prepro.subheader("Après tokenisation est padding nous obtenons :")
embedding=tokenize_text(texte)
prepro.write(embedding)
prepro.text("")

#  classif
classif.text("")
classif.text("")
classif.subheader("Architecture du meilleur modèle")
path_modele='archi_model.png'
classif.image(path_modele,  use_column_width=True,caption=["BiLSTM 100 units"])



# Show subprocess and progress
classe=""
with st.spinner("Prédiction en cours..."):
    if st.button("Predire la classe du produit"):
        classe=predict(embedding)
st.success('La classe de ce produit est  {}'.format(classe))


