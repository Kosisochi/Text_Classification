import pandas as pd
import numpy as np
#import gensim
import time
import tensorflow as tf
import random as rn
# np.random.seed(12)
# rn.seed(12)
# tf.random.set_seed(12)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, concatenate, Concatenate, Input, LSTM, Bidirectional
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,classification_report
import tensorflow
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


#LOAD DATA
df1 = pd.read_csv('/home/xxxxxxxxxxxxxxx/Train.csv', encoding='latin1',lineterminator='\n')
df1['cleantext'] = df1['cleantext'].astype(str)
train_x = df1['cleantext'].values.tolist()
train_y = df1['HS'].values

#train_val split
train_x, val_x, train_y, val_y = train_test_split(train_x,train_y, random_state=12, test_size=0.15)

df2 = pd.read_csv('/home/xxxxxxxxxxxxxxxxxx/Dev.csv', encoding='latin1',lineterminator='\n')
df2['cleantext'] = df2['cleantext'].astype(str)
test_x = df2['cleantext'].values.tolist()
test_y = df2['HS'].values


#tokenize, convert to sequence and pad sequences
tokenizer= Tokenizer()
tokenizer.fit_on_texts(train_x)
train_sequences = tokenizer.texts_to_sequences(train_x)
train_sequences = pad_sequences(train_sequences, maxlen = 60, padding = 'post')
val_sequences = tokenizer.texts_to_sequences(val_x)
val_sequences = pad_sequences(val_sequences, maxlen = 60, padding = 'post')
test_sequences = tokenizer.texts_to_sequences(test_x)
test_sequences = pad_sequences(test_sequences, maxlen = 60, padding = 'post')
word_index = tokenizer.word_index
num_words = len(word_index)
vocab_size = num_words + 1
embeddings_index = {}
f = open(os.path.join('','em-glove.6B.300d-20epoch'), encoding = 'utf-8')
for line in f:                          #for every line in the saved embedding,
    values = line.split()               #split that line and load it into 'values'

    word = values[0]                    #pick the first value in values ie the value with index 0 and load it into 'word'
    coefs = np.asarray(values[1:])      #create an array of the remaining values in values from index 1 to the end and load it into 'coefs'
    embeddings_index[word] = coefs      #then load word and coefs in to the dictionary 'embeddings_index'
f.close()

vocab_size = num_words + 1  #this should be equal to the length of word_index.items
print(vocab_size)
embedding_matrix = np.zeros((vocab_size, 300))

for word, index in  word_index.items():   #word_index is a dictionary containg the words after tokenization and their index. the index is assigned based on the frequency of occurence of each word. Therefore, the word with index 1 will be the word with the highest frequency of occurence.
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word) #this gets the vector of the word
        if embedding_vector is not None:             #if that vector isn't empty
            embedding_matrix[index] = embedding_vector  #add the the index and the vector to the matrix


w2v_matrix = tf.keras.utils.normalize(embedding_matrix, axis=-1, order=2)
w2v_matrix1 = np.nan_to_num(w2v_matrix)


accuracy_scores = []
recall_scores = []
precision_scores=[]
f1_scores =[]
AverageF1_forPnN = []

for i in range(30):
    B = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    tensorflow.random.set_seed(B[i])
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=60, weights=[embedding_matrix],trainable=False))
    model.add(Bidirectional(LSTM(100, dropout=0.2)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', baseline=0.4, patience=80, verbose=1)
    mc = ModelCheckpoint('saved_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    hist = model.fit(train_sequences, train_y, epochs=100, verbose=False, batch_size=100,
                      validation_data=(val_sequences, val_y),
                     callbacks=[es, mc])
    # load the saved model
    saved_model = load_model('saved_model.h5')


    pred_prob = saved_model.predict(test_sequences)  # predict probabilities for the test set
    trainresults = saved_model.evaluate(train_sequences, train_y)
    print('Train Set Evaluation results:', trainresults)
    testresults = saved_model.evaluate(test_sequences, test_y)
    print('Test Set Evaluation results:', testresults)

    y_pred = np.argmax(pred_prob, axis = 1)

    #calculate Precision, recall and F1
    accuracy = accuracy_score(test_y, y_pred)
    accuracy_scores.append(accuracy)
    print('Accuracy: %f' % accuracy)

    precision = precision_score(test_y, y_pred, average='macro')
    precision_scores.append(precision)
    print('Precision: %f' % precision)

    recall = recall_score(test_y, y_pred, average='macro')
    recall_scores.append(recall)
    print('Recall: %f' % recall)

    f1 = f1_score(test_y, y_pred, average='macro')
    f1_scores.append(f1)
    print('F1 score: %f' % f1)

    cnf_matrix = confusion_matrix(test_y, y_pred, labels=[0, 1])
    print(cnf_matrix)
    target_names = ['class 0', 'class 1']
    print(classification_report(test_y, y_pred, target_names=target_names))


mean_accuracy = np.mean(np.array(accuracy_scores))
mean_precision = np.mean(np.array(precision_scores))
mean_recall = np.mean(np.array(recall_scores))
mean_f1 = np.mean(np.array(f1_scores))

std_dev_accuracy = np.std(np.array(accuracy_scores), axis=0, dtype=np.float64)
std_dev_precision = np.std(np.array(precision_scores), axis=0, dtype=np.float64)
std_dev_recall = np.std(np.array(recall_scores), axis=0, dtype=np.float64)
std_dev_f1 = np.std(np.array(f1_scores), axis=0, dtype=np.float64)

print("Mean Accuracy",mean_accuracy,std_dev_accuracy)
print("Mean Precision",mean_precision, std_dev_precision)
print("Mean Recall",mean_recall, std_dev_recall)
print("Mean F1",mean_f1, std_dev_f1)

print("Accuracy list", accuracy_scores)
print("Precision list", precision_scores)
print("Recall list", recall_scores)
print("F1 list", f1_scores)

