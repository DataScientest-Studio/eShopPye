# ================================ IMPORTS ====================================

# Srtreamlit
import streamlit as st

# Essential
import numpy as np
import pandas as pd

# System
import os
import time

# Image
from PIL import Image
import cv2

# Deep learning
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16





# def write():
#     """Used to write the page in the app.py file"""
    
# =============================== CONSTANTS ===================================

LABELS = range(27)

# ================================= PATHS =====================================

# Get this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set paths
PATH_DECODE = SCRIPT_DIR + os.path.normpath("/data/csv/class_index_decoding.csv")
PATH_PRODUCT = SCRIPT_DIR + os.path.normpath("/data/csv/prdtype_decode_enhanced.csv")
PATH_DUMMY = SCRIPT_DIR + os.path.normpath("/assets/images/imgdummy.png")
PATH_MODEL = SCRIPT_DIR + os.path.normpath("/data/models/vgg16_fullset_v2.h5")

# ================================ CODE =======================================











st.title("Classification des images")
st.header("Choix de l'image")
img = cv2.imread(r"C:\Users\basti\ANACONDA\eShopPye\bastien\streamlit\pages\classification_images\assets\images\test_img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgslot1 = st.image(img/255, caption="Image sélectionnée")

# img_io = getdata.file_uploader( # returns UploadedFile (subclass of BytesIO)
#                     label = "Choisissez une image à tester:",
#                     type = ['png', 'jpg', 'jpeg'],
#                     accept_multiple_files = False
#                     )
# # Create image slot and put placeholder there
# dummy_io = Image.open(fullpath) # read image in BytesIO format
# imgslot1 = getdata.image(dummy_io)

# if img_io:
#     imgslot1.image(img_io, caption="Selected image")
#     img = np.array(img_io)


st.header("Prétraitement")
imgslot2 = st.empty()
# Preprocess
#if img_io and img:
img_resz = cv2.resize(img, (256, 256))
img_prepro = vgg16.preprocess_input(img_resz)
imgslot2.image(np.clip(img_prepro/255, 0, 1), caption="Image prétraitée")


st.header("Chargement du modèle")
model = load_model(PATH_MODEL)


st.header("Classification")
gopredict = st.button("Lancer la prédiction")
result = st.spinner("En attente de la prédiction...")
if gopredict:
    with st.spinner("Classification en cours..."):
        # Predict probabilities
        y_probs = model.predict(np.reshape(img_prepro, (1, 256, 256, 3))).squeeze()
        # Decode class indexes into labels
        decoding = pd.read_csv(PATH_DECODE, index_col='label').sort_index().squeeze()
        # Reorder probabilities (according to labels)
        y_probs = [y_probs[decoding[label]] for label in LABELS]
        # Decision
        y_pred = np.argmax(y_probs)
        # Find product definition
        products = pd.read_csv(PATH_PRODUCT, index_col='label')['prdtype'].squeeze()
        # Display
        st.success(f"Classe prédite : [{y_pred}] '{products[y_pred]}'")
        # Emulate heavy computing (for user experience!)
        time.sleep(0.1)


