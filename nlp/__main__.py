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


def retrieve_card_text(card_ids = []): # this is cleared of issues
    """
    This function retrieves the oracle text from the scryfall API for a list of card ids in the training set.

    Args: 
        card_ids: A list of card ids to retrieve the oracle text for.
    """
    base_url = "https://api.scryfall.com/cards/"
    card_ids = []

    with open("nlp/training_and_testing_set.csv", "r", newline="") as csv_file:
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

def preprocess_text(cards_text = list):# this is cleared of issues
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
    processed_text = [" ".join(card) for card in processed_text]

    return processed_text

def final_data_prep(processed_text: list = None, testing_and_training_set = csv):
    """ 
    This function prepares the input dataset for NLP training.

    Args:
        processed_text: A list of processed card text to train the model on.
        testing_and_training_set: The CSV file containing the training and testing set (combined).
    """

    processed_text = preprocess_text()
    data = pd.read_csv("nlp/training_and_testing_set.csv")
    data.insert(loc = 0, column = "text", value = processed_text)
    data.drop(columns = ["id"], inplace = True)
    text = data["text"].values
    labels = data.loc[:, data.columns != "text"].values
    train_text, test_text, train_labels, test_labels = train_test_split(text, labels, test_size=0.4)

    return data, train_text, test_text, train_labels, test_labels

def laplace_smoothing(n_text_with_label, vocab, word_counts_dict, word, text_label):
    """
    This function compensates for words that are presnt in the test set and not in the training set by returning smoothed condiditonal probabilities.

    Args:
        n_label_items:
        vocab: unique words in dataset
        word_counts_dict: dictionary of unique words and their frequency in dataset
        word: missing word
        text_label: label assoicated with missing words???????
    """
    a = word_counts_dict[text_label][word] + 1
    b = n_text_with_label[text_label] + len(vocab)
    return math.log(a/b)
    
def grouping_labels(x, y, labels):
    """
    This function groups data by label dependant on positive values.

    Args:
        x: text (values)
        y: labels (values)
        labels: labels (headers)
    Returns:
        data: groups of text by positive associated label
    """ 
    data = {}
    for label in labels:
        data[label] = x[np.where(y == 1)]
    return data    

def fit(x, y, labels):
    """
    This function takes x (text values) and y (labels values) and returns the number of cards with each label and its apriori conditional probability.

    Args:
        x: text (values)
        y: labels (values)
        labels: labels (headers)
    Returns:
        n_text_with_label: number of card texts associated with each label
        log_label_probs: log of the apriori conditional probablities
    """
    n_text_with_label = {}
    log_label_probs = {}
    n = len(x)
    grouped_data = grouping_labels(x, y, labels)
    for i, data in grouped_data.items():
        n_text_with_label[i] = len(data)
        log_label_probs[i] = math.log(n_text_with_label[i] / n)
    return n_text_with_label, log_label_probs

def predict(n_text_with_label, vocab, word_counts_dict, log_label_probs, labels, x):
    """
    This function returns the predictions by the nlp model on how fed in card text should be calssified

    Args:
        n_text_with_label: number of card texts associated with each label
        vocab: unique words in dataset
        word_counts_dict: dictionary of unique words and their frequency in dataset
        log_label_probs: log of the apriori conditional probablities
        labels: labels (headers)
        x: text (values)
    """
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    result = []
    for text in x:
        label_scores = {i: log_label_probs[i] for i in labels}
        words = set(tokenizer.tokenize(text))
        for word in words:
            if word not in vocab: continue
            for i in labels:
                log_w_given_i = laplace_smoothing(n_text_with_label, vocab, word_counts_dict, word, 1)
                label_scores[i] += log_w_given_i
        result.append(max(label_scores, key=label_scores.get))
    return result


if __name__ == "__main__":
    
    data, train_text, test_text, train_labels, test_labels = final_data_prep()
    vec = CountVectorizer(max_features=3000)
    x = vec.fit_transform(data)
    vocab = vec.get_feature_names_out()
    x = x.toarray()
    word_counts_dict = {}
    for i in range(len(vocab)):
        word_counts_dict[i] = defaultdict(lambda: 0) 
    print(word_counts_dict)
    for j in range(x.shape[0]):
        i = train_labels[j]
        for k in range(len(vocab)):
            word_counts_dict[i][vocab[k]] += x[j][k] 
    
    labels = [0,1]
    n_text_with_label, log_label_probs = fit(train_text, train_labels, labels)
    prediction = predict(n_text_with_label, vocab, word_counts_dict, log_label_probs, labels, test_text)
    print("Accuraccy of prediction for test set: ", accuracy_score(test_labels, prediction))
