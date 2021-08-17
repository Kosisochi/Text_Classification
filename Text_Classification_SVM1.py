#SVM for char n gram 

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
from sklearn.metrics import classification_report, confusion_matrix
from itertools import product
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

##load dataset
df1 = pd.read_csv('Dataset.csv', encoding='latin1')
df1['train_tweet'] = df1['train_tweet'].astype(str)
tweets = df1['train_tweet']
tweet_list = df1['train_tweet'].values.tolist()
y= df1['train_label'].values

###train-test split
train_x, test_x, train_y, test_y = model_selection.train_test_split(df1['train_tweet'], y, random_state=12)


#concatinating the train and test input in other to fit the vectorizer on it
all_text = pd.concat([train_x, test_x])

#creating a vectorized char ngram
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char_wb',
    stop_words='english',
    ngram_range=(2,5),
    max_features=5000)
char_vectorizer.fit(all_text)
train_features_dr = char_vectorizer.transform(train_x) #output is a csr matrix
test_features_dr = char_vectorizer.transform(test_x) #output is a csr matrix


print('Word feature',char_vectorizer.get_feature_names())


# #FEATURE SELECTION
# print("1: ", train_features.shape)
# print("1a: ",test_features.shape)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectFromModel
#
# select = SelectFromModel(LogisticRegression(class_weight='balanced', penalty = 'l2', C=0.01, solver= 'lbfgs'), threshold= '1.25*mean')
# train_features_dr = select.fit_transform(train_features,train_y)
# test_features_dr = select.transform(test_features)
#
# print("2: ",train_features_dr.shape)
# print("2a: ",test_features_dr.shape)

# #GRID SEARCH
# parameter_candidates = [
#
#     {'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [1000], 'class_weight' :[{0:1, 1:1, 2:1},{0:20,1:1, 2:1}, {0:10, 1:1, 2:1}, 'balanced']},
#     {'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [10000], 'class_weight' :[{0:1, 1:1, 2:1},{0:20,1:1, 2:1}, {0:10, 1:1, 2:1}, 'balanced']},
#     {'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [100000], 'class_weight' :[{0:1, 1:1, 2:1},{0:20,1:1, 2:1}, {0:10, 1:1, 2:1}, 'balanced']},
#
# ]
# # Create a classifier object with the classifier and parameter candidates
#
# grid_search = GridSearchCV(estimator=svm.LinearSVC(), param_grid=parameter_candidates, scoring= 'recall_macro' , cv = 5,  n_jobs=-1)
# # Train the classifier on data1's feature and target data
# grid_search.fit(train_features_dr, train_y)
# # View the accuracy score
# print('Best score for train_features:', grid_search.best_score_)
# # View the best parameters for the model found using grid search
# print('Best C:',grid_search.best_estimator_.C)  #for svm.LinearSVC
# print('Best max-iter:',grid_search.best_estimator_.max_iter)  #for svm.LinearSVC
# print('Best max-iter:',grid_search.best_estimator_.class_weight)  #for svm.LinearSVC


#BUILD THE MODEL, FIT AND PREDICT
SVM_for_CNgram = svm.LinearSVC(C = 0.1  ,multi_class = 'ovr', class_weight='balanced', penalty= 'l2', max_iter=1000)
SVM_for_CNgram.fit(train_features_dr, train_y)
predictions = SVM_for_CNgram.predict(test_features_dr)
#pred_classes =SVM_for_WCNgram.predict_classes(test_features)   #predict crisp classes for the test set
#these prediction returned in a 2D array


# NB_for_WCNgram = naive_bayes.MultinomialNB() #WCNgram is word and char n gram
# NB_for_WCNgram.fit(train_features,train_y)
# predictions = NB_for_WCNgram.predict(test_features)
# print("Accuracy for WCNgram Features and NB Classifier:",metrics.accuracy_score(predictions, test_y))


print("Accuracy for CNgram Features and SVM Classifier:", metrics.accuracy_score(test_y,predictions))
print("Macro Precision for CNgram Features and SVM Classifier:", metrics.precision_score(test_y, predictions, average='macro',labels= [0,1,2], pos_label=0))
print("Macro Recall for CNgram Features and SVM Classifier:", metrics.recall_score(test_y, predictions, average='macro' , labels= [0,1,2], pos_label=0))
print("Macro F1 score for CNgram Features and SVM Classifier:", metrics.f1_score(test_y, predictions, average='macro' , labels= [0,1,2], pos_label=0))



print(confusion_matrix(test_y, predictions))
print(classification_report(test_y, predictions))

#Function to print the instances classified as False positive and negative
def check_predictions(predictions,truth):
    # take a 1-dim array of predictions from a model, and a 1-dim truth vector and calculate similarity
    # returns the indices of the false negatives and false positives in the predictions.

    truth=truth.astype(bool)
    predictions=predictions.astype(bool)
    print (sum(predictions == truth), 'of ', len(truth), "or ", float(sum(predictions == truth))/float(len(truth))," match")

    # false positives
    print ("false positives: ", sum(predictions & ~truth))
    # false negatives
    print ("false negatives: ",sum( ~predictions & truth))
    false_neg=np.nonzero(~predictions & truth) # these are tuples of arrays
    false_pos=np.nonzero(predictions & ~truth)
    return false_neg[0], false_pos[0] # we just want the arrays to return

# get the indices for false_negatives and false_positives in the test set
false_neg, false_pos= check_predictions(test_y, predictions)

# map the false negative indices in the test set (which is features) back to it's original data (text)
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

