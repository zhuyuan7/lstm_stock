# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import Embedding # 수정
from keras.preprocessing import sequence
# from keras_preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences # 수정
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tf_data import TF_Data
from tensorflow import keras
import pandas
# from keras.models import load_model
from os.path import isfile
import os
from keras.optimizers import Adam
from tensorflow.python.eager.context import get_config
import tensorflow as tf
import pickle



def train(filename, model_name, day='tomorrow'):
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(model_name, monitor='val_acc', mode='auto', verbose=1, save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1)
    df = pandas.read_csv(filename)
    print(df)
    headlines = df['normalized_headline']
    # print(headlines)
    top_words = 2000

    data = TF_Data(filename, top_words=top_words)
    pickle.dump(data, open(filename.replace(".csv", ".p"), "wb"))
    # load the dataset but only keep the top n words, zero the rest
    (X_train, y_train), (X_test, y_test) = data.load_data(day=day)
    print(X_train.shape)

    # truncate and pad input sequences
    max_review_length = 100
    # X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    # X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    X_train = pad_sequences(X_train, maxlen=max_review_length) # 수정
    X_test = pad_sequences(X_test, maxlen=max_review_length)   # 수정
    # create the model
    embedding_vecor_length = 32
    
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    
    
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    if isfile(model_name) and False:
        print('Checkpoint Loaded')
        model = save_model(model_name)
    print(model.summary())

    model.fit(X_train.astype(int), y_train.astype(int), validation_split=0.02, epochs=1, batch_size=64,
              callbacks=[checkpoint, earlyStopping, reduceLR]) # .astype(int) 추가 
    
    # Final evaluation of the model
    #test data model evaluation 
    # -> best model 
    # save 
    # model = load_model(model_name)
    
    # model = tf.keras.models.load_model(model_name) #수정
    
    scores = model.evaluate(X_test.astype(int), y_test.astype(int), verbose=0)
    fd = open('accuracy.csv', 'a')
    CsvRow = [filename, day, "Accuracy: %.2f%%" % (scores[1] * 100)]
    print(CsvRow)
    fd.write(", ".join(CsvRow) + "\n")
    fd.close()
    #Saving the model
    file_name = 'save_model/{}.h5'.format(day)
    model.save(file_name)
    print("Saved model `{}` to disk".format(file_name))

for f in os.listdir(r"/home/joo/Stock-Market-Prediction/data/"):
    if (f.endswith('.csv')):
        for d in ['ms_today', 'ms_tomorrow']:
            train(os.path.join(r"/home/joo/Stock-Market-Prediction/data/", f), f.replace(".csv", "_" + d + "_.hdf5"), d)

