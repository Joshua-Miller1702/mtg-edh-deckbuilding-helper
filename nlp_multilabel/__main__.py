import pandas as pd
import json
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.corpus
import csv
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def retrieve_card_text(): #Works
    """
    This function retrieves the oracle text from the training set to convert to a list for preprocessing.
    
    Returns: 
        cards_text - oracle text of cards
    """
    df = pd.read_csv("data/train_validate.csv")
    df.drop(df.loc[df["waste"] == 1].index, inplace = True)
    cards_text = df["oracle_text"].values

    return cards_text
    
def preprocess_text(cards_text: list = None): #Works
    """
    This function preprocesses the card text in perpartion for NLP training.

    Args:
        cards_text - A list of cards text to process.
    Returns:
        processed_text - preprocessed text to pass to be added to dataframe to act as feature vectors for the model.
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
    
    Args:
        processed_text - oracle text that has been preprocessed for addition to dataframe.
    Returns:
        data - dataframe containting all data needed to train nlp model.
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
        X_train - preprocessed text for train set
        X_validate - preprocessed text for validation set
        labels - list of label column headers
        train - train portion of entire dataset
        validate - validate portion of entire dataset 
    """
    data = data if data else complete_dataset()

    data = pd.DataFrame(data)

    label_values = (data.loc[:, data.columns != "processed_text"].values)
    text_data = data["processed_text"].values
    labels = data.columns
    labels = labels[1:]

    train, validate = train_test_split(data, test_size = 0.25, random_state = 17)
    X_train = train.processed_text
    X_validate = validate.processed_text

    return X_train, X_validate, labels, train, validate

def look_at_data():
    """
    This function creates a plot of the dataset used to train the model.
    """
    data = complete_dataset()
    data = pd.DataFrame(data)
    data.plot(subplots=True)
    plt.tight_layout()
    plt.show()

def pipelines(X_train, X_val, labels, meta_train, meta_validate):
    NB_pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer()),
                    ("classifier", OneVsRestClassifier(MultinomialNB(fit_prior = True, class_prior = None)))
                    ])

    for label in labels:
        NB_pipeline.fit(X_train, meta_train[label])
        prediction = NB_pipeline.predict(X_val)
        print(f"Accuracy for {label} is: {accuracy_score(meta_validate[label], prediction)}")


if __name__ == "__main__":
    X_train, X_val, labels, meta_train, meta_validate = data_prep_alt()
    pipelines(X_train, X_val, labels, meta_train, meta_validate)