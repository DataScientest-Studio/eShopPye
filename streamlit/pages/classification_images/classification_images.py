# ================================ IMPORTS ====================================
import streamlit as st
import awesome_streamlit as ast

import pandas as pd
import numpy as np
import os
import shutil
import cv2
from PIL import Image
import requests # to get image from the web
import shutil # to save it locally
import inspect


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    
# =============================== CONSTANTS ===================================

    # Récupérer le chemin de ce script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Ouvrir une image au format BytesIO
    #IMGDUMMY = Image.open(SCRIPT_DIR + r"\assets\images\imgdummy.png")
    IMGDUMMY = Image.open(SCRIPT_DIR + os.path.normpath("/assets/images/imgdummy.png"))
    

# ================================ CODE =======================================

# ------------------------------- LAYOUT --------------------------------------

    
    st.title("Classification des images")
    st.header("Choix de l'image")
    getdata = st.beta_container()
    st.header("Prétraitement")
    prepro = st.beta_container()
    st.header("Classification")
    classif = st.beta_container()


# ------------------------------- BACKEND -------------------------------------

    # CONTAINER: getdata
    img = getdata.file_uploader(
                        label = "Choisissez une image à tester:",
                        type = ['png', 'jpg', 'jpeg'],
                        accept_multiple_files = False
                        )
    imgslot = getdata.image(IMGDUMMY)
    if img:
        imgslot.image(img)
    
    
    # CONTAINER: prepro
    prepro.text("Ceci est le conteneur 'prepro'")
    
    # CONTAINER: classif
    classif.text("Ceci est le conteneur 'classif'")
    
    # Display block of code (and execute it)
    with st.echo():
        a = 2 # do stuff
    
    # Show subprocess and progress
    with st.spinner("Prédiction en cours..."):
        pass # do stuff
    st.success("Prédiction terminée.")