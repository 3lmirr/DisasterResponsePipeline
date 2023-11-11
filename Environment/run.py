import json
import plotly
import pandas as pd
import re
import nltk
import sklearn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download(['punkt','stopwords'])
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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


stemmer = PorterStemmer()

#Tokenization function for CountVectorizer
def series_tokenizer(pd_series): 
    wt = word_tokenize(re.sub(r'[^a-zA-Z]',' ',pd_series.lower()))
    stemmed_series = [stemmer.stem(i) for i in wt if i not in stopwords.words('english')]
    return stemmed_series

vect = CountVectorizer(tokenizer = series_tokenizer)
tfidf = TfidfTransformer()


#Normalization function for Word Counter
def word_normalize(text):
    reg = [re.sub(r'[^a-zA-Z]', " ", z.lower()) for z in text]
    token = [word_tokenize(i) for i in reg]
    stem = [[stemmer.stem(i) for i in x if i not in stopwords.words('english')] for x in token]
    final = [" ".join(i) for i in stem]
    
    return final

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


app = Flask(__name__)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

# load data
engine = create_engine("sqlite:///data/DisasterPipeline.db")
df = pd.read_sql_table('DisasterTable', engine)

# load model
model = joblib.load("app/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()