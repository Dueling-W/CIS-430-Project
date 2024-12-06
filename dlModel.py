# Importing libraries

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import matplotlib
import tensorflow as tf
from tensorflow import keras
import sklearn
import spacy
from num2words import num2words
import re


def textPreprocessing(text):
    

    #1. lowercase text, create doc object
    text = text.lower()
    doc = nlp(text)

    #2. filter out stop words
    no_stop = [token.text for token in doc if not token.is_stop]
    processed = ' '.join(no_stop)
    doc = nlp(processed)

    #3. filter out puncucation
    no_punc = [token.text for token in doc if not token.is_punct]
    processed = ' '.join(no_punc)
    doc = nlp(processed)

    #4. no single characters
    no_single = [token.text for token in doc if not len(token)==1]
    processed = ' '.join(no_single)
    doc = nlp(processed)

    #5. lemminization
    lemm_tokens = [token.lemma_ for token in doc]
    processed = ' '. join(lemm_tokens)

    #6. num to words using regular expression
    def replace_numbers(match):
        num = int(match.group())
        return num2words(num)
    
    processed_text = re.sub(r'\b\d+\b', replace_numbers, processed)

    return processed_text

def classPercentages(df):

    class_counts = df['labels'].value_counts()
    class_percentages = df['labels'].value_counts(normalize=True) * 100

    print(f'Class Counts: {class_counts}\n Class Percentages: {class_percentages}')



nlp = spacy.load("en_core_web_sm")
dataset = 'bbc_data_pre_processed.csv'
df = pd.read_csv(dataset)

print(df)






