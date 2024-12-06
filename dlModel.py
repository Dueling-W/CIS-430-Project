# Import functions

import preProcessing as func
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import InputLayer
from keras.optimizers import Adam
from sklearn.preprocessing import OrdinalEncoder


def createModel(unique_words, max_len):

    opt = Adam(learning_rate=0.003)
    output_dim = 50
    model = Sequential()
    print(f'Creating embed layer using {unique_words} words, {max_len} length, output dim of {output_dim}')

    input = InputLayer(input_shape = (max_len,))
    model.add(input)
    embed = Embedding(input_dim = unique_words, output_dim=output_dim, trainable=True)
    model.add(embed)
    model.add(LSTM(100))
    model.add(Dense(1, activation='tanh'))

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

    

if __name__ == "__main__":
    dataset = 'bbc_data_pre_processed.csv'
    df = pd.read_csv(dataset)
    count = func.returnCount(df)
    data_train, data_test, labels_train, labels_test, length = func.mlPreprocessing(df)

    model = createModel(count, length)
    print(model.summary())

    #model.fit(data_train, labels_train, epochs=20, validation_split=0.2, verbose=2)










