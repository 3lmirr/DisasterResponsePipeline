import sys
import pandas as pd
import numpy as np
import re
import sklearn
import sqlalchemy
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download(['punkt','stopwords'])
import pickle
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

database_filepath  = "data\DisasterPipeline.db"
messages_filepath = "data\disaster_messages.csv"
categories_filepath = "data\disaster_categories.csv"

def load_from_db(database_filepath):

    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterResponse",engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1)
    category_names = y.columns

    return X,y, category_names

#Normalization function for Word Counter
def word_normalize(text):
    reg = [re.sub(r'[^a-zA-Z]', " ", z.lower()) for z in text]
    token = [word_tokenize(i) for i in reg]
    stem = [[stemmer.stem(i) for i in x if i not in stopwords.words('english')] for x in token]
    final = [" ".join(i) for i in stem]
    
    return final

#Analyzing frequency of words to create categories as a new feature
def word_counter(text):
    sentences = word_normalize(text)
    joined_text = ' '.join(sentences)
    tokenized_words = joined_text.split()
    word_counts = Counter(tokenized_words)
    
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_word_counts:
        print(f'{word}: {count}')


#Tokenization function for CountVectorizer
def series_tokenizer(pd_series): 
    wt = word_tokenize(re.sub(r'[^a-zA-Z]',' ',pd_series.lower()))
    stemmed_series = [stemmer.stem(i) for i in wt if i not in stopwords.words('english')]
    return stemmed_series

tfidf = TfidfTransformer()
stemmer = PorterStemmer()
vect = CountVectorizer(tokenizer = series_tokenizer)

#Creating Category class for generating new feature other than vectorized data 
class Category(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.category_list = ['water','food','earthquak','flood','rain','tent','aid','storm','diseas','hurrican'
                           ,'medic','river','tsunami','drought','cyclon','fire','wind','snow','ebola','malaria'
                           ,'mosquito','hurricanesandi']
        self.df = pd.DataFrame({name:[] for name in self.category_list})
        
    def fit(self,X,y=None):
        return self
    
    
    def define_category(self,text):
        text = pd.Series(word_normalize(text))
        concatted_df = pd.concat([text,self.df],axis=1)
        for category in self.category_list:
            for row in range(len(text)):
                if category in text[row]:
                    concatted_df[category].iloc[row] = '1'
                else:
                    concatted_df[category].iloc[row] = '0'
        concatted_df.drop(0, axis=1, inplace=True)
        concatted_df = concatted_df.astype('int') 
        return concatted_df      
    
    def transform(self, X):
        concatted_df = self.define_category(X)
        category_matrix = csr_matrix(concatted_df) #Converting df into matrix to match with the text pipeline results
        return category_matrix


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=series_tokenizer)),
                ('tfidf', TfidfTransformer())
            ])),
                #New column processing class for category
                ('category', Category()) 
        ])),
        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])
    
    param_grid = {
            'clf__estimator__learning_rate': [0.01, 0.1],
            'clf__estimator__n_estimators': [100, 250],
            'clf__estimator__max_depth': [3, 5],
            'clf__estimator__base_score': [0.5, 0.75]  
    }
    
    cv = GridSearchCV(pipeline, param_grid)
    return cv
    
    
def evaluate_model(y_train, X_pred,y_test, y_pred):
    
    train_classification_matrix = {}
    train_acc_matrix = {}
    
    
    test_classification_matrix = {}
    test_acc_matrix = {}
    
    for i in range(len(y_train.columns)):
        y_true_train = y_train.iloc[:,i]
        y_train_predicted = X_pred[:,i]
        
        train_report = classification_report(y_true_train,y_train_predicted, zero_division=1)
        train_accuracy = accuracy_score(y_true_train,y_train_predicted)
        
        train_classification_matrix[y_train.columns[i]] = train_report
        train_acc_matrix[y_train.columns[i]] = train_accuracy
        
    for column, report in train_classification_matrix.items():
        print(f"Classification Report for {column}:\n{report}\n")
        
    for column, report in train_acc_matrix.items():
        print(f"Accuracy Report for {column}:\n{report}\n")
        
    for i in range(len(y_test.columns)):
        y_test_true = y_test.iloc[:,i]
        y__test_predicted = y_pred[:,i]
        
        test_report = classification_report(y_test_true,y__test_predicted, zero_division=1)
        test_accuracy = accuracy_score(y_test_true,y__test_predicted)

        test_acc_matrix[y_test.columns[i]] = test_accuracy
        test_classification_matrix[y_test.columns[i]] = test_report
        
    for column, report in test_classification_matrix.items():
        print(f"Classification Report for {column}:\n{report}\n")
    
    for column, report in test_acc_matrix.items():
        print(f"Accuracy Report for {column}:\n{report}\n")
    
    
def save_model(model):
    with open('model.pkl','wb') as model_file:
        pickle.dump(model,model_file)
        


def main():
    if len(sys.argv) == 2:
        database_filepath = sys.argv[1:]
        X, y, category_names = load_from_db(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        print('Building model...')
        model = build_model()
        
        start_time = time.time()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        end_time = time.time()
        fit_time_seconds = end_time - start_time
        fit_time_minutes, fit_time_seconds = divmod(fit_time_seconds, 60)
        fit_time_hours, fit_time_minutes = divmod(fit_time_minutes, 60)
        print(f'Fit time: {int(fit_time_hours)} hours and {int(fit_time_minutes)} minutes')
        
        best_params = model.best_params_
        print(f"Best parameters: {best_params}")
        model.estimator.set_params(**best_params)
        X_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        print('Evaluating model...')
        evaluate_model(y_train, X_pred,y_test, y_pred)

        print('Saving model...')
        save_model(model)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')



if __name__ == '__main__':
    main()