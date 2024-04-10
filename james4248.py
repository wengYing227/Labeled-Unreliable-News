import pandas as pd
import numpy as np
import time
import csv
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from gensim.models import KeyedVectors
from scipy.sparse import hstack
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

start_time = time.time()

wordVec = KeyedVectors.load("word2vec-google-news-300.model", mmap='r')

def textLen(text):
    lengths = []

    for t in text:
        count = len(t.split())
        lengths.append([count])
    
    return np.array(lengths).reshape(-1, 1)

def meanEmbed(text, wordVec):
    mean = []
    
    for t in text:
        wordVecs = []

        for word in t.split():
            if word in wordVec:
                wordVecs.append(wordVec[word])
    
        if not wordVecs:
            wordVecs.append(np.zeros(wordVec.vector_size))

        tempmean = np.mean(wordVecs, axis=0)
        mean.append(tempmean)

    return np.array(mean)

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vect', TfidfVectorizer(ngram_range=(1,1), max_df=0.9, min_df=2, max_features=30)),
            # ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)), 
        ])),
        ('length', FunctionTransformer(textLen)),
        # ('wordvec', FunctionTransformer(meanEmbed, kw_args={'wordVec': wordVec})),
    ])),
    ('clf', LogisticRegression(max_iter=1000))
])


def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    model.fit(X_train, y_train)

def predict(model, X_test):
    ''' TODO: make your prediction here '''
    return model.predict(X_test)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    csv.field_size_limit(999999)
    train = pd.read_csv('raw_data/fulltrain.csv', header = None, names=['class','text'])
    X_train = train['text']
    y_train = train['class']
    model = pipeline

    train_model(model, X_train, y_train)
    # test your model

    # generate prediction on test data
    test = pd.read_csv("raw_data/balancedtest.csv", header = None, names=['class','text'])
    X_test = test['text']
    y_test = test['class']

    y_pred = predict(model, X_test)

    # Use f1-macro as the metric
    score = f1_score(y_test, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    print('classification report on test data')
    print(classification_report(y_test, predict(model, X_test)))


# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
    end_time = time.time()
    print('time: {}'.format(end_time - start_time))
