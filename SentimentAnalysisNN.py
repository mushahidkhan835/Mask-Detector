import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
import seaborn as sn
import nltk
from tensorflow.keras.layers import Dropout
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from pathlib import Path
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import plot_confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

class DataPreProcessing:
    
    def __init__(self, df):
        self.df = df
        self.model = None
        
    def remove_stopwords(self, text):
        list_of_stop_words = stopwords.words('english')
        whitelist = ["n't", "not", "no"]
        words = text.split() 
        clean_words = [word for word in words if (word not in list_of_stop_words or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words)
    
    def preprocess_data(self):
        self.df = self.df.loc[:, ('text', 'airline_sentiment')]
        self.df.text = self.df.text.apply(lambda text: re.sub(r'@\w+', '', text))
        self.df.text = self.df.text.apply(self.remove_stopwords)
        self.df.dropna()
   
    def get_df(self):
        return self.df
    
    def convert_target_class_to_numbers(self, y_train, y_test):
        label_encoder = LabelEncoder()
        y_train_converted = to_categorical(label_encoder.fit_transform(y_train))
        y_test_converted = to_categorical(label_encoder.transform(y_test))
        return (y_train_converted, y_test_converted)
    
    def get_df(self):
        return self.df
    
    def create_tokenizer(self, x, number_of_unique_words):
        tokenizer = Tokenizer(num_words=number_of_unique_words,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ")
        tokenizer.fit_on_texts(x)
        return tokenizer
    

class NeuralNetwork:

    def __init__(self, number_of_epochs, max_len, number_of_dimensions, number_of_unique_words):
        self.number_of_epochs = number_of_epochs  
        self.max_len = max_len
        self.number_of_dimensions = number_of_dimensions
        self.number_of_unique_words = number_of_unique_words
        
    
    def create_model(self):
        model = models.Sequential()
        model.add(layers.Embedding(self.number_of_unique_words, self.number_of_dimensions, input_length=self.max_len))
        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))
        self.model = model
        return model
        
    def compile_and_fit(self, X_train, y_train, X_valid=None, y_valid=None,):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model.fit(X_train
                         , y_train
                         , epochs=self.number_of_epochs
                         , validation_split=0.2
                         , batch_size=32
                         , verbose=0)
    def predict_classes(self, x):
        return np.argmax(self.model.predict(x), axis=-1)
    
    def test_model(self, X_test, y_test):     
        return self.model.evaluate(X_test, y_test, verbose=0)
        
    def get_model(self):
        return self.model
    
def plot_metrics(history, metric_name, model_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, len(history.history[metric_name]) + 1)

    plt.plot(e, metric, '-bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, '-go', label='Validation ' + metric_name)
    plt.title('Training Data ' + ' & ' + 'Validation Data Accuracy Using NN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()

    
def run_experiment(train_path):
    #Get data frame
    df = pd.read_csv(train_path)

    #Preprocess data
    data_preprocessing = DataPreProcessing(df)
    data_preprocessing.preprocess_data()
    df = data_preprocessing.get_df()

    #Split data
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.2, random_state=20)  

    #Convert target class to matrix class
    y_train, y_test = data_preprocessing.convert_target_class_to_numbers(y_train, y_test)
      
    #Tokenize training and test data
    number_of_words = 0
    longest_sentence = ''
    all_words = []
    
    for row in X_train:
        tokenize_word = word_tokenize(row)
        if len(tokenize_word) > len(longest_sentence):
            longest_sentence = tokenize_word
        for word in tokenize_word:
            all_words.append(word)

    number_of_unique_words = len(set(all_words))

    tokenizer = data_preprocessing.create_tokenizer(X_train, number_of_unique_words)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    #Create word sequences of equal length
    pad_sequences(X_train_seq, maxlen=len(longest_sentence))
    X_train_seq = pad_sequences(X_train_seq, maxlen=len(longest_sentence))
    X_test_seq =  pad_sequences(X_test_seq, maxlen=len(longest_sentence))
    
    itera = 1
    acc = 0
    neural_network = NeuralNetwork(4, len(longest_sentence), 100, number_of_unique_words)

    #Neural Network with Word Embedding
    neural_network.create_model()

    history = neural_network.compile_and_fit(X_train_seq, y_train)

    #Evaluate model
    plot_metrics(history, 'accuracy', 'Neural Network')

    loss, accuracy = neural_network.model.evaluate(X_test_seq, y_test, verbose=0)

    print('Test accuracy of our model: {0:.2f}%'.format(accuracy * 100))

    y_pred = neural_network.predict_classes(X_test_seq)


    y_test_numerical_values = []
    for val in y_test:
        if val[0] == 1:
            y_test_numerical_values.append(0)
        elif val[1] == 1:
            y_test_numerical_values.append(1)
        elif val[2] == 1:
            y_test_numerical_values.append(2)

    print(metrics.classification_report(y_test_numerical_values, y_pred, digits=3))

    sn.heatmap(metrics.confusion_matrix(y_test_numerical_values, y_pred), annot=True,cmap='Blues', fmt='g')

def main():
    return run_experiment('Tweets.csv')
