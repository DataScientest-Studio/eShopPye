# Imports
import re
import string
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# NTLK corpuses
nltk.download('stopwords')
nltk.download('punkt')
STOPWORDS_FR = nltk.corpus.stopwords.words('french')

# Other
PONCTUATION = string.punctuation


# VERSION
version = 1


def get_word_count(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len(str(x).split())



def get_unique_words(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len(set(str(x).split()))



def get_stop_words_count(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len([w for w in str(x).lower().split() if w in STOPWORDS_FR])



def get_chart_count(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len(str(x))



def get_mean_word_length(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len(set(str(x).split()))



def get_punctuation_count(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len([c for c in str(x) if c in PONCTUATION])
        

def remplace_accent(x):
    if pd.isnull(x):
        return x
        
    accent = ['é', 'è', 'ê', 'à', 'ù', 'û', 'ç', 'ô', 'î', 'ï', 'â','&acirc;','&agrave;','&eacute;',
              '&ecirc;','&egrave;','&euml;','&icirc;','&iuml;','&ocirc;','&oelig;','&ucirc;','&ugrave;','&uuml;',
              '&ccedil;','&lt;','&gt;','&szlig;','&oslash;','&Omega;','&ETH;','&Oslash;','&THORN;','&thorn;','&Aring;']
    sans_accent = ['e', 'e', 'e', 'a', 'u', 'u', 'c', 'o', 'i', 'i', 'a','a','a','e','e','e','e','i','i','o','oe','u',
                   'u','u','c',' ',' ','',' ',' ',' ',' ',' ',' ','A']

    for c, s in zip(accent, sans_accent):
        x = x.replace(c, s)
        
    return x

    

def cleaning_data(x):

    if pd.isnull(x):
        return x
 
    sentences = nltk.sent_tokenize(x)
    stemmer =  nltk.stem.snowball.FrenchStemmer()

    # Lemmatization
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [stemmer.stem(word) for word in words if word not in STOPWORDS_FR]
        sentences[i] = ' '.join(words) 
    sentences=' '.join(sentences)
    
    # On retire les balises html
    sentences = re.sub('<[^<]+?>', '', sentences)
   
    return sentences
    
    
    
def corpus(x,y):

    if pd.isnull(x):
        return y
    elif pd.isnull(y):
        return x
    else :
        return x+y
        

def plot_ngrams(dict_ngram, size=50, title="Décompte des n-grammes", color='b', figsize=(16,9)):

    keys = list(dict_ngram.keys())
    vals = [dict_ngram[k] for k in keys]
    ngrams = pd.DataFrame({'ngrams': keys, 'count': vals})
    ngrams = ngrams.sort_values(['count'], ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=figsize)
    sns.barplot(y=ngrams['ngrams'][:size], x=ngrams['count'][:size], color=color)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.xlabel("count", fontsize=15)
    plt.ylabel("n-grams", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show();
    
    
def generate_ngrams(text, n_gram=1):

    stopwords = STOPWORDS_FR + ['&','#',';','a','e'] + [str(i) for i in range(32, 48)]
    token = [token for token in text.lower().split(' ') if token != '' if token not in stopwords]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    stg = [' '.join(ngram) for ngram in ngrams]
    
    return stg
    

def plot_ngrams_class(ngrams_dict,  size=10, title='n-grams', subplots=(14,2), color='b', figsize=(16,70)):
    fig = plt.figure(figsize=figsize)

    sub_idx = 1
    for i, key in enumerate(list(ngrams_dict.keys())):
        keys = list(ngrams_dict[key].keys())
        vals = [ngrams_dict[key][mot] for mot in keys]
        grams_df = pd.DataFrame(columns=['grams', 'count'])
        grams_df['grams'] = keys
        grams_df['count'] = vals
        grams_df = grams_df.sort_values(['count'], ascending=False).reset_index(drop=True)
        plt.subplot(subplots[0], subplots[1], sub_idx)
        sns.barplot(y=grams_df['grams'][:size], x=grams_df['count'][:size], color=color)
        plt.title(title + "pour la classe {}".format(key))
        sub_idx += 1