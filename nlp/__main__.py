import re
import pandas as pd
import numpy as np
import requests
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

def retrieve_card_text(card_ids = []):
    """
    This function retrieves the oracle text from the scryfall API for a list of card ids in the training set.

    Args: 
        card_ids: A list of card ids to retrieve the oracle text for.
    """
    base_url = "https://api.scryfall.com/cards/"
    card_ids = []

    with open("nlp/training_set.csv", "r", newline="") as csv_file:
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

def preprocess_text(cards_text = list):
    """
    This function preprocesses the card text in perpartion for NLP training.

    Args:
        cards_text: A list of cards text to process.
    """
    cards_text = retrieve_card_text()

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
        
    return processed_text

if __name__ == "__main__":
    processed_text = preprocess_text()
    print(processed_text)
#make nlp
#pass csv into nlp
#nlp will grab card info, separtate the text and then then sort into bukcets
