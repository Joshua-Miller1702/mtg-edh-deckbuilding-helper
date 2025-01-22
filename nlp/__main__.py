import re
import pandas as pd
import numpy as np
import requests
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import math
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict


def retrieve_card_text(card_ids: list = None):
    """
    This function retrieves the oracle text from the scryfall API for a list of card ids in the training set.

    Args: 
        card_ids: A list of card ids to retrieve the oracle text for.
    """
    base_url = "https://api.scryfall.com/cards/"
    card_ids = card_ids if card_ids else []

    with open("nlp/train_test.csv", "r", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter = ",")
        for row in reader:
            card_ids.append(row[0])
    card_ids.pop(0)

    cards_text = []
    for card_id in card_ids:
        url = base_url + card_id
        response = requests.get(url).json()
        key = "oracle_text"
        cards_text.append(response.get(key))

    return cards_text

def preprocess_text(cards_text: list = None):
    """
    This function preprocesses the card text in perpartion for NLP training.

    Args:
        cards_text: A list of cards text to process.
    """
    cards_text = cards_text if cards_text else retrieve_card_text()

    for i in range(len(cards_text)):
        cards_text[i] = re.sub(r'\d+', '', cards_text[i])
        cards_text[i] = re.sub(r'[^\w\s]', '', cards_text[i])
        cards_text[i] = re.sub(r'\n', ' ', cards_text[i])
        cards_text[i] = cards_text[i].lower()

    nltk.download('stopwords')
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

def final_data_prep(processed_text: list = None, train_test: csv = None):
    """ 
    This function prepares the input dataset for NLP training.

    Args:
        processed_text: A list of processed card text to train the model on.
        train_test: The CSV file containing the training and testing set (combined).
    """
    processed_text = processed_text if processed_text else preprocess_text()
    data = pd.read_csv(train_test) if train_test else pd.read_csv("nlp/train_test.csv")
    data.insert(loc = 0, column = "text", value = processed_text)
    data.drop(columns = ["id"], inplace = True)
    text = data["text"].values
    labels = data.loc[:, data.columns != "text"].values
    train_text, test_text, train_labels, test_labels = train_test_split(text, labels, test_size=0.4)

    return data, train_text, test_text, train_labels, test_labels

if __name__ == "__main__":
    
    data, train_text, test_text, train_labels, test_labels = final_data_prep(train_test = "nlp/train_test.csv")
