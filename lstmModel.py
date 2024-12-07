# Import functions

import preProcessing as func
import pandas as pd
import tensorflow as tf
import sys
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

    # Higher value = more complicated model (probably better performance, but could be overkill)
    # Smaller value = less complicated model (could be better or worse performance)
    output_dim = 32

    # Sigmoid seems to be the best activation function, but you can try other ones (list here: https://keras.io/api/layers/activations/)
    activation = 'sigmoid'

    # Create model
    model = Sequential()
    print(f'Creating embed layer using {unique_words} words, {max_len} length, output dim of {output_dim}')

    # Input layer equal to length of sequences 
    input = InputLayer(input_shape = (max_len,))
    model.add(input)

    # Creating dense vector of words (floating point values)
    embed = Embedding(input_dim = unique_words, output_dim=output_dim, trainable=True)
    model.add(embed)

    # LSTM layer (main logic), can tweak number of units
    model.add(LSTM(units=40, activation=activation))

    # Dropout layer (removes some units), can also be tweaked
    model.add(Dropout(0.1))

    # Final layer, shouldn't be tweaked
    model.add(Dense(5, activation=activation))

    # Compile with Adam, learning rate can be tweaked
    opt = Adam(learning_rate=0.003)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

    

if __name__ == "__main__":

    # Pre-process data, return testing and training sets
    dataset = 'bbc_data_pre_processed.csv'
    df = pd.read_csv(dataset)
    count = func.returnCount(df)
    data_train, data_test, labels_train, labels_test, length = func.mlPreprocessing(df)

    # Creat model, print out summary of the overall layers
    model = createModel(count, length)
    print(model.summary())

    # Stops model at perfect point, can change the patience value
    early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

    # Train the model, epochs can be changed but mostly uncessary with early stopping
    model.fit(data_train, labels_train, epochs=100, validation_split=0.2, verbose=2, callbacks=[early_stop])

    










