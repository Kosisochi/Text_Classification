#using bilstm as a fine-tuning layer after the Encoder-12-FeedForward-Norm layer.
#This cuurently used the whole matrix including the representations for CLS and SEP as input matrix. We need to experiment with sentence matrix without the CLS and SEP representation

import os
import codecs
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from chardet import detect
from sklearn.model_selection import train_test_split
import time
import gc
import keras
from keras_radam import RAdam
from keras import backend as K
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#PARAMETERS
SEQ_LEN = 37
BATCH_SIZE = 100
EPOCHS = 10
LR = 1e-4

#Path to the pre trained model of BERT.
pretrained_path = '/xxxxxxxxxxxxxx/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, '/xxxxxxxxxxxxxx/uncased_L-12_H-768_A-12/bert_config.json')
checkpoint_path = os.path.join(pretrained_path, '/xxxxxxxxxxxxxx/uncased_L-12_H-768_A-12/bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, '/xxxxxxxxxxxxxx/uncased_L-12_H-768_A-12/vocab.txt')



#Loading Pretrained BERT model.
model = load_trained_model_from_checkpoint(config_path,checkpoint_path,training=True,trainable=True,seq_len=SEQ_LEN)
#model.summary()

#Extracting token dictionary from vocab of pretrained model to refer for input we will be using.
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

#Defining Tokenizer.
tokenizer = Tokenizer(token_dict)


DATA_COLUMN = 'train_tweet'
LABEL_COLUMN = 'train_label'


#Load and Convert to data that BERT understands
def convert_data(data_df):
    global tokenizer
    indices, targets = [], []
    for i in tqdm(range(len(data_df))):
        ids, segments = tokenizer.encode(data_df[DATA_COLUMN][i], max_len=SEQ_LEN)
        indices.append(ids)
        targets.append(data_df[LABEL_COLUMN][i])
    items = list(zip(indices, targets))
    np.random.shuffle(items)
    indices, targets = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(targets)
def load_data(path):
    data_df = pd.read_csv(path, nrows=18587)
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y

train_x, train_y = load_data('train_tweet.csv')
gc.collect()

t0=time.time()
#Extracting layer from pretrained bert model and adding a layer with softmax function to classify 3 classes.
inputs = model.inputs[:2]  #We are getting all layers EXCEPT last 2 layers
layer_output = model.get_layer('Encoder-12-FeedForward-Norm').output  # (?, 37, 768)    #NSP-Dense is the first dense layer after the output of [CLS] token.
#print(layer_output.get_shape())
#input_shape = keras.layers.Input(shape=(SEQ_LEN,768))(layer_output)
bilstm = keras.layers.Bidirectional(keras.layers.LSTM(768, dropout=0.2, recurrent_dropout=0.2, input_shape=(SEQ_LEN,768), return_sequences=False))(layer_output) #(input_shape)
outputs = keras.layers.Dense(units=3, activation='softmax')(bilstm)
model = keras.models.Model(inputs, outputs)
model.compile(RAdam(learning_rate =LR),loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.summary()

print ("model building  time:", round(time.time()-t0, 3), 's')
#Initializing variables.
sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables])
sess.run(init_op)

t1=time.time()
#Finally, training the model.
model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE)
print ("model training time:", round(time.time()-t1, 3), 's')

DATA_COLUMN = 'test_tweet'
LABEL_COLUMN = 'test_label'
#Load the test data
def convert_test(test_df):
    global tokenizer
    indices, targets = [], []
    for i in tqdm(range(len(test_df))):
        ids, segments = tokenizer.encode(test_df[DATA_COLUMN][i], max_len=SEQ_LEN)
        indices.append(ids)
        targets.append(test_df[LABEL_COLUMN][i])
    items = list(zip(indices, targets))
    np.random.shuffle(items)
    indices, targets = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(targets)

def load_test(path):
    data_df = pd.read_csv(path, nrows=6196)
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y

test_x, test_y= load_test('test_tweet.csv')
gc.collect()

t2=time.time()
#Making prediction for test dataset.
predicts = model.predict(test_x, verbose=True).argmax(axis=-1)
print ("model predicting time:", round(time.time()-t2, 3), 's')

#Calculating accuracy.
print(np.sum(test_y == predicts) / test_y.shape[0])

#y_pred = np.argmax(predicts, axis = 1)
accuracy = accuracy_score(test_y, predicts)
precision = precision_score(test_y, predicts, average='macro')
recall = recall_score(test_y, predicts, average='macro')
f1_macro = f1_score(test_y, predicts, average='macro')
f1_weighted= f1_score(test_y, predicts, average='weighted')
f1_micro = f1_score(test_y, predicts, average='micro')
cnf_matrix = confusion_matrix(test_y,predicts, labels=[0, 1, 2])
print('accuracy', accuracy)
print('precision', precision)
print('recall', recall)
print('f1', f1_macro)
print('f1-weighted', f1_weighted)
print('f1-micro', f1_micro)
print(cnf_matrix)