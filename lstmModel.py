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
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import OrdinalEncoder


def createModel(unique_words, max_len):

    output_dim = 32
    model = Sequential()
    print(f'Creating embed layer using {unique_words} words, {max_len} length, output dim of {output_dim}')

    input = InputLayer(input_shape = (max_len,))
    model.add(input)
    embed = Embedding(input_dim = unique_words, output_dim=output_dim, trainable=True)
    model.add(embed)
    model.add(LSTM(40))
    model.add(Dropout(0.1))
    model.add(Dense(5, activation='sigmoid'))

    opt = Adam(learning_rate=0.003)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

    

if __name__ == "__main__":
    dataset = 'bbc_data_pre_processed.csv'
    df = pd.read_csv(dataset)
    count = func.returnCount(df)
    data_train, data_test, labels_train, labels_test, length = func.mlPreprocessing(df)

    model = createModel(count, length)
    print(model.summary())

    early_stop = EarlyStopping(monitor="val_accuracy", patience=10)


    model.fit(data_train, labels_train, epochs=100, validation_split=0.2, verbose=2, callbacks=[early_stop])









