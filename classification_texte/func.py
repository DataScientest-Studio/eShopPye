# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:39:00 2021

@author: 33664
"""

import tensorflow as tf

import re, unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('french'))
from nltk.stem.snowball import FrenchStemmer
import pandas as pd




@tf.function
def macro_soft_f1(y, y_hat):
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    # on multiplie la proba prédite d'une classe (y_hat) par son label => Uniquement les probas des vrais positifs seront non nuls
    tp = tf.reduce_sum(y_hat * y, axis=0) 
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    #  calcul du F1 score , 1e-16 pour ne pas diviser par 0
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    # comme on cherche a maximiser F1_score , et qu'il nous faut une fonction coût à minimiser on calcul le cout= 1 - soft-f1 
    cost = 1 - soft_f1 
    # on fait la moyenne pour tous les labels du batch
    macro_cost = tf.reduce_mean(cost) 
    return macro_cost

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1



def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
    
def remplace_accent(x):
    cd=pd.isnull(x)
    if cd==True:
        return x
    accent = ['é', 'è', 'ê', 'à', 'ù', 'û', 'ç', 'ô', 'î', 'ï', 'â','&acirc;','&agrave;','&eacute;','&ecirc;','&egrave;','&euml;','&icirc;','&iuml;','&ocirc;','&oelig;','&ucirc;','&ugrave;','&uuml;','&ccedil;','&lt;','&gt;','&szlig;','&oslash;','&Omega;','&ETH;','&Oslash;','&THORN;','&thorn;','&Aring;']
    sans_accent = ['e', 'e', 'e', 'a', 'u', 'u', 'c', 'o', 'i', 'i', 'a','a','a','e','e','e','e','i','i','o','oe','u','u','u','c',' ',' ','',' ',' ',' ',' ',' ',' ','A']
    for c, s in zip(accent, sans_accent):
        x = x.replace(c, s)
        return x

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w=remplace_accent(w)
    #removing html tags
    w=re.sub('<[^<]+?>', '', w)
    # creating a space between a word and the punctuation following it
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    #Remove words of length less than 1
    w=re.sub(r'\b\w{,1}\b', '', w)
     # remove stopword
    mots = word_tokenize(w.strip())
    stemmer =  FrenchStemmer()
    mots = [stemmer.stem(mot) for mot in mots if mot not in stop_words]
    return ' '.join(mots).strip()