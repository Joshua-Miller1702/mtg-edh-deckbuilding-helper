import pandas as pd
import json
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk.corpus
import csv
from torch.utils.data import TensorDataset, DataLoader
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score






def retrieve_card_text(): #Works
    """
    This function retrieves the oracle text from the training set to convert to a list for preprocessing.
    """
    df = pd.read_csv("data/train_validate.csv")
    df.drop(df.loc[df["waste"] == 1].index, inplace = True)
    cards_text = df["oracle_text"].values

    return cards_text
    
def preprocess_text(cards_text: list = None): #Works
    """
    This function preprocesses the card text in perpartion for NLP training.

    Args:
        cards_text: A list of cards text to process.
    """
    cards_text = cards_text if cards_text else retrieve_card_text()

    for i in range(len(cards_text)):
        cards_text[i] = re.sub(r'\d+', '', str(cards_text[i]))
        cards_text[i] = re.sub(r'[^\w\s]', '', str(cards_text[i]))
        cards_text[i] = re.sub(r'\n', ' ', str(cards_text[i]))
        cards_text[i] = str(cards_text[i]).lower()

    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    processed_text = []
    for card in cards_text:
        tokenized = tokenizer.tokenize(card)
        stopword_removal = [word for word in tokenized if word not in stopwords]
        lemmatized = [lemmatizer.lemmatize(word) for word in stopword_removal]
        processed_text.append(lemmatized)
    processed_text = [" ".join(card) for card in processed_text]

    return processed_text

def complete_dataset(processed_text: list = None): #Works
    """
    This function combines the processed text with the csv data containing the labels.
    """
    processed_text = processed_text if processed_text else preprocess_text()
    data = pd.read_csv("data/train_validate.csv")
    data.drop(columns = "oracle_id", inplace = True)
    data.drop(columns = "oracle_text", inplace = True)
    data.drop(data.loc[data["waste"] == 1].index, inplace = True)
    data.drop(columns = "waste", inplace = True)
    data.drop(columns = "mdfc", inplace = True)
    data.drop(columns = "poison", inplace = True)
    data.drop(columns = "vehicle", inplace = True)
    data.drop(columns = "damage_multiplyers", inplace = True)
    data.insert(loc = 0, column = "processed_text", value = processed_text)

    return data

def data_prep_alt(data: csv = None):
    """ 
    This function prepares the input dataset for NLP training.

    Args: 
        data - text and lables to pass in as a .csv file.
    Returns: 
        labels - list of label column headers
        X_train_dataset - dataset containing vectorised words to train the model on.
        X_validate_dataset - dataset containing vectorised words to validate the model on.
    """
    data = data if data else complete_dataset()

    data = pd.DataFrame(data)

    label_values = (data.loc[:, data.columns != "processed_text"].values)
    text_data = data["processed_text"].values
    labels = data.columns
    labels = labels[1:]

    X_train, X_validate, Y_train, Y_validate = train_test_split(text_data, label_values, test_size = 0.25, random_state = 17)

    vectorizer = CountVectorizer()
    X_train_fit = vectorizer.fit_transform(X_train)
    X_validate_fit = vectorizer.transform(X_validate) #fit to vectorizer not model?

    return X_train_fit, Y_train, X_validate_fit, Y_validate

if __name__ == "__main__":
    X, Y, X_val, Y_val = data_prep_alt()
    model = LabelPowerset(GaussianNB())
    model.fit(X, Y)
    predictions = model.predict(X_val)
    print(accuracy_score(Y_val,predictions))    
    