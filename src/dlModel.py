# Import functions

import preProcessing as func
import pandas as pd
import tensorflow as tf
import sys
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import InputLayer
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt


def createModel(unique_words, max_len, tag):

    # Higher value = more complicated model (probably better performance, but could be overkill)
    # Smaller value = less complicated model (could be better or worse performance)
    output_dim = 32

    # Sigmoid seems to be the best activation function, but you can try other ones (list here: https://keras.io/api/layers/activations/)
    activation = 'sigmoid'

    # Create model
    model = Sequential()
    print(f'\nCreating embed layer using {unique_words} words, {max_len} length, output dim of {output_dim}\n')

    # Input layer equal to length of sequences 
    input = InputLayer(input_shape = (max_len,))
    model.add(input)

    # Creating dense vector of words (floating point values)
    embed = Embedding(input_dim = unique_words, output_dim=output_dim, trainable=True)
    model.add(embed)

    # LSTM layer (main logic), can tweak number of units
    model.add(Bidirectional(LSTM(units=40, activation=activation)))

    # Dropout layer (removes some units), can also be tweaked
    model.add(Dropout(0.1))

    if(tag=='bbc'):
        # Final layer, shouldn't be tweaked
        model.add(Dense(5, activation=activation))
    else:
        model.add(Dense(1, activation=activation))

    # Compile with Adam, learning rate can be tweaked
    opt = Adam(learning_rate=0.0031)

    if(tag=='bbc'):
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


    return model


def generateGraph(history):

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['train', 'validation'], loc='upper left')


    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(['train', 'validation'], loc='upper left')

    fig.suptitle('Bidirectional LSTM Model Training')
    plt.show()

    

if __name__ == "__main__":

    # Process dataset
    dataset = 'data/spam_data_pre_processed.csv'
    df = pd.read_csv(dataset)

    # Return # of unique words, create test/train datasets
    count = func.returnCount(df, 'data')
    data_train, data_test, labels_train, labels_test, length = func.mlPreprocessing(df, 'data')

    # Creat model, print out summary of the overall layers
    tag = 'spam' # spam or bbc
    model = createModel(count, length, tag)
    print(model.summary())

    # Stops model at perfect point, can change the patience value
    early_stop = EarlyStopping(monitor="val_accuracy", patience=12)
    #lr_monitor = ReduceLROnPlateau(monitor='val_loss', factor=.7, patience=3)

    # Train the model, epochs can be changed but mostly uncessary with early stopping
    history = model.fit(data_train, labels_train, epochs=100, validation_split=0.2, verbose=2, callbacks=[early_stop])

    results = model.evaluate(data_test, labels_test)
    print(f'\nEvaluated Loss: {results[0]}; Evaluated Accuracy: {results[1]}')

    generateGraph(history)



    










