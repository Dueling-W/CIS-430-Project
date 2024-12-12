# Developing Recurrent Neural Networks for Text Classification: A Comparative Study

### **Project Introduction**

CIS 430 Project in Fall 2024 created by: William Girard and Cameron Pacheco. Investigation into recurrent neural networks and text classification.
***
### **Project Description**
- Identified text classification datasets
- Performed text preprocessing (removing stop words, etc.)
- Performed deep learning text preprocessing using Tensorflow's Keras
- Created recurrent neural networks (RNNs) for the purpose of multiclass and binary class text classification
- Compared four RNNs: Bidirectional LSTM, GRU, LSTM, and SimpleRNN
- Use evaluation metrics to compare the RNNs
***
### **Description of GitHub Project**
- src contains main files
  - dlModel.py is the main program which runs the deep learning models
  - preProcessing.py contains functions to be called by dlModel.py for text preprocessing, doesn't run on its own
- data contains all .csv files used in this project
- ml_images contain all matplotlib graphs of the pre-training process, also shown in the paper
***
### **Running the Project**
- dlModel.py will start the training and testing process of our Bidirectional LSTM model by simply running it
- Can change between datasets by setting the tag variable, in line 96, to either bbc or spam
- bbc corresponds to the BBC text classification dataset and spam corresponds to the spam dataset
