import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import matplotlib
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import spacy
from num2words import num2words
import re
from collections import Counter

from sklearn.model_selection import train_test_split



def textPreprocessing(df, label):

    data = df[label]

    for (text, i) in zip(data, range(0, len(df))):
        
        print(f'\n{i}/{len(df)}: Data being processed...')
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

        df.loc[i, [label]] = processed_text
        print(f'{i}/{len(df)}: Data finished processing!\n')

    return df

def classPercentages(df):

    class_counts = df['labels'].value_counts()
    class_percentages = df['labels'].value_counts(normalize=True) * 100

    print(f'Class Counts: {class_counts}\n Class Percentages: {class_percentages}')


def returnCount(df):

    data = df['data']
    count = Counter()

    for text in data:
        split_text = text.split()
        for word in split_text:
            count[word] += 1
    
    return int(len(count))

def mlPreprocessing(df):

    data = df.iloc[:, 0]  
    labels = df.iloc[:, -1]  

    data_train, data_test, labels_train, labels_test = train_test_split(data.to_numpy(), labels.to_numpy(), train_size=.80, random_state=42)

    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
    count = returnCount(df)
    token = Tokenizer(num_words=count)
    token.fit_on_texts(data_train)

    # https://stackoverflow.com/questions/42943291/what-does-keras-io-preprocessing-sequence-pad-sequences-do

    train_seq = token.texts_to_sequences(data_train)
    test_seq = token.texts_to_sequences(data_test)

    sequence_lengths = [len(seq) for seq in train_seq]
    #print("90th percentile length:", np.percentile(sequence_lengths, 95))
    #print("Max length:", max(sequence_lengths))

    length = 350

    train_padded = pad_sequences(train_seq, maxlen=length, padding='post', truncating='post', value = 0.0)
    test_padded = pad_sequences(test_seq, maxlen=length, padding='post', truncating='post', value = 0.0)

    return train_padded, test_padded, labels_train, labels_test, length



if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")