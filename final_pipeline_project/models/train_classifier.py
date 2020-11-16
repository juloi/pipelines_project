import sys
import sqlite3 as sq
import re
import pickle

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk

nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stopwords_english = stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    Loads data from an SQLITE3 database and return predictor and target variables
    :param database_filepath: filepath to an SQLITE3 database file
    :return:
        X: predictor variables
        y: target variables
        category_names: the names of the target variables
    '''

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages_df', con=engine, schema='main', index_col='id')
    X = df['message']
    y = df.loc[:, "related":]
    category_names = y.columns.to_list()
    return X, y, category_names


def tokenize(text):
    '''
    Function to be used as a custom tokenizer in TfidfVectorizer
    :param text: a single text message
    :return: a list of nornalized, lemmatized tokens
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = tok.strip()
        if clean_tok in stopwords_english:
            continue
        else:
            lemma = lemmatizer.lemmatize(clean_tok, pos='v')
            clean_tokens.append(lemma)

    return clean_tokens


def build_model():
    """
    Prepares the GridSearchCV object based on a pipeline and
    parameter grid
    :return: GridSearchCV object
    """
    pipeline = Pipeline([
        ('tf', TfidfVectorizer()),
        ('multi_clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        'tf__tokenizer': [tokenize],
        'tf__max_df': [1.0, 0.95],
        'tf__use_idf': [True, False],
        'multi_clf__estimator__n_jobs': [-1],
        'multi_clf__estimator__n_estimators': [10, 20, 100]
    }

    search = GridSearchCV(pipeline, param_grid=parameters)

    return search


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the classification model
    :param model: fitted GridSearchCV object
    :param X_test: test predictor data set
    :param Y_test: test target data set
    :param category_names: names of the target variables
    :return: prints out Precision, Recall and F-1 metrics for each of the target variables
    """
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        col_preds = y_pred[:, i]
        col_test = Y_test[col]
        print(col)
        print(classification_report(col_test, col_preds))


def save_model(model, model_filepath):
    """
    Pickles moden in binary into a file.
    :param model: final classification model
    :param model_filepath: directory to store the pickeled model
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
