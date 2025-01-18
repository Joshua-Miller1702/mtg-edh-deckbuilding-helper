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

def retrieve_card_text(card_ids = []):
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
    processed_text = [" ".join(card) for card in processed_text]

    return processed_text

def final_data_prep(processed_text = list, testing_and_training_set = csv):
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
    lables = data.loc[:, data.columns != "text"].values
    train_text, test_text, train_labels, test_labels = train_test_split(text, lables, test_size=0.4)

    return data, train_text, test_text, train_labels, test_labels



def nlp_model(data = pd.DataFrame):
    """
    This function trains an NLP model on the processed card text.

    Args:
        processed_text: A list of processed card text to train the model on.
    """
    data, train_text, test_text, train_labels, test_labels = final_data_prep()
    vec = CountVectorizer() #Unique words
    vocab = vec.get_feature_names_out() #get unique words out?
    x = vec.fit_transform(data) #fits data into model and transforms data to suit the model?
    x = x.toarray()
    word_counts_dict = {}
    for i in range(37): # !!!!! probably not 2 for me xd !!!!!!# but may be change 
        word_counts_dict[i] = defaultdict(lambda: 0) #dict that dosent throw keyerror, lambda fills it with "key": 0?
    for j in range(x.shape[0]):
        i = train_labels[j]
        for k in range(len(vocab)):
            word_counts_dict[i][vocab[k]] += x[j][k]

def laplace_smoothing(n_label_items, vocab, word_counts_dict, word, text_label):
    """
    This function compensates for words that are presnt in the test set and not in the training set by returning smoothed condiditonal probabilities.

    Args:
        n_label_items:
        vocab: unique words in dataset
        word_counts_dict: dictionary of unique words and their frequency in dataset
        word: ??????
        text_label: ?????
    """
    a = word_counts_dict[text_label][word] + 1
    b = n_label_items[text_label] + len(vocab)
    return math.log(a/b)
    
def grouping_labels(x, y, labels):
    """
    This function groups data by label dependant on positive values.

    Args:
        x: text (values)
        y: labels (values)
        labels: labels (headers)
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
    """








if __name__ == "__main__":
    processed_text = preprocess_text()
    data = final_data_prep(processed_text)
    print(processed_text)
    "nlp_model(processed_text)"
    print(data)
#make nlp
#pass csv into nlp
#nlp will grab card info, separtate the text and then then sort into bukcets
