#SVM for word n gram

import pandas as pd
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import svm
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from itertools import product
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score

##Load Dataset
df1 = pd.read_csv('Dataset.csv', encoding='latin1')
df1['train_tweet'] = df1['train_tweet'].astype(str)
tweets = df1['train_tweet']
tweet_list = df1['train_tweet'].values.tolist()
y= df1['train_label'].values

###train-test split
train_x, test_x, train_y, test_y = model_selection.train_test_split(df1['train_tweet'], y, random_state=12)

#concatenating the train and test input in other to fit the vectorizer on it
all_text = pd.concat([train_x, test_x])

#creating a vectorized word unigram
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    stop_words='english',
    ngram_range=(1, 3),
    max_features=10000)
word_vectorizer.fit(all_text)
train_features = word_vectorizer.transform(train_x)#output is a csr matrix
test_features = word_vectorizer.transform(test_x) #output is a csr matrix
#print(test_features)

#print('Word feature',word_vectorizer.get_feature_names())


#FEATURE SELECTION
print("1: ", train_features.shape)
print("1a: ",test_features.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

select = SelectFromModel(LogisticRegression(class_weight='balanced', penalty = 'l2', C=0.01, solver= 'lbfgs'), threshold= '1.25*mean')
#select = SelectFromModel(svm.LinearSVC(C=0.1, multi_class='ovr', class_weight='balanced', penalty='l2', max_iter=1000), threshold= '1.25*mean')
train_features_dr = select.fit_transform(train_features,train_y)
test_features_dr = select.transform(test_features)

print("2: ",train_features_dr.shape)
print("2a: ",test_features_dr.shape)

###The hyperparameters for this SVM was found using Grid Search
SVM_for_WNgram = svm.LinearSVC(C = 0.1 ,multi_class = 'ovr', class_weight='balanced', penalty= 'l2', max_iter=1000)
SVM_for_WNgram.fit(train_features_dr, train_y)
y_pred = SVM_for_WNgram.predict(test_features_dr)

# NB_for_WCNgram = naive_bayes.MultinomialNB() #WCNgram is word and char n gram
# NB_for_WCNgram.fit(train_features,train_y)
# predictions = NB_for_WCNgram.predict(test_features)
# print("Accuracy for WCNgram Features and NB Classifier:",metrics.accuracy_score(predictions, test_y))


print("Accuracy for CNgram Features and SVM Classifier :", metrics.accuracy_score(test_y, y_pred))
print("Macro Precision for CNgram Features and SVM Classifier :", metrics.precision_score(test_y, y_pred, average='macro'))
print("Macro Recall for CNgram Features and SVM Classifier :", metrics.recall_score(test_y, y_pred, average='macro'))
print("Macro F1 score for CNgram Features and SVM Classifier :", metrics.f1_score(test_y, y_pred, average='macro' ))

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred,labels=[0, 1, 2]))


##Function to print the instances classified as False positive and negative
def check_predictions(predictions,truth):
    # take a 1-dim array of predictions from a model, and a 1-dim truth vector and calculate similarity
    # returns the indices of the false negatives and false positives in the predictions.

    truth=truth.astype(bool) # converts to bool
    predictions=predictions.astype(bool) # converts to bool
    print (sum(predictions == truth), 'of ', len(truth), "or ", float(sum(predictions == truth))/float(len(truth))," match")

    # false positives
    print ("false positives: ", sum(predictions & ~truth))
    # false negatives
    print ("false negatives: ",sum( ~predictions & truth))

    false_neg=np.nonzero(~predictions & truth) # these are tuples of arrays
    false_pos=np.nonzero(predictions & ~truth)
    return false_neg[0], false_pos[0] # we just want the arrays to return

# get the indices for false_negatives and false_positives in the test set
false_neg, false_pos= check_predictions(test_y, y_pred)

# # map the false negative indices in the test set (which is features) back to it's original data (text)
# print ("False negatives: \n")
# pd.options.display.max_colwidth = 140
# for i in false_neg:
#     original_index=idx_test[i]
#     print (df['index3'].iloc[original_index], df['tweet2'].iloc[original_index])
#
# print("False positive: \n")
# pd.options.display.max_colwidth = 140
# for i in false_pos:
#     original_index = idx_test[i]
#     print(df['index3'].iloc[original_index],df["tweet2"].iloc[original_index])
