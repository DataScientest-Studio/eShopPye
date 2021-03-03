# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:33:00 2021

@author: basti
"""

# ================================ IMPORTS ====================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import cv2
from PIL import Image
import requests # to get image from the web
import shutil # to save it locally

# =============================== CONSTANTS ===================================

PATH_DOWNLOADS = os.path.abspath("./downloads")
PATH_PLACEHOLDER = r"C:\Users\basti\ANACONDA\eShopPye\bastien\streamlit\assets\placeholder_image.png"
    # NB: not ideal but only this works...
PLACEHOLDER_IMG = Image.open(PATH_PLACEHOLDER)

# ================================= CODE ======================================

# ---------------------------- INITIALIZATION ---------------------------------



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
imgslot = getdata.image(PLACEHOLDER_IMG)
if img:
    imgslot.image(img)


# CONTAINER: prepro
prepro.text("Ceci est le conteneur 'prepro'")

# CONTAINER: classif
classif.text("Ceci est le conteneur 'classif'")





