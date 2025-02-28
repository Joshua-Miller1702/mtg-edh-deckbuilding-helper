import pandas as pd
import json
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk.corpus
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.adapt import MLkNN
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
import pickle


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
    data.drop(data.loc[data["waste"] == 1].index, inplace = True)
    data.drop(columns = ["oracle_id", 
                         "oracle_text", 
                         "waste", 
                         "mdfc", 
                         "poison", 
                         "alternate_win", 
                         "vehicle", 
                         "damage_multiplyers",   
                         "cheat", 
                         "flexible", 
                         "jank"], inplace = True)
    data.insert(loc = 0, column = "processed_text", value = processed_text)

    return data

def data_prep_dummy():
    """
    This function generates a dummy dataset to test that the model is working as intended.
    """
    data = complete_dataset()
    df = pd.DataFrame(data)
    text_data = df["processed_text"].values
    newdf = df.replace(1,0)
    label_values = (newdf.loc[:, newdf.columns != "processed_text"].values)
    labels = newdf.columns
    labels = labels[1:]

    X_train, X_val, Y_train, Y_val = train_test_split(text_data, label_values, test_size=0.25, random_state=17)

    return X_train, X_val, labels, Y_train, Y_val

    
    l = 72

def data_prep_main(data: csv = None):
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
    labels = data.loc[:, data.columns !="preprocessed_text"]
    X_train, X_val, Y_train, Y_val = train_test_split(text_data, label_values, test_size=0.25, random_state=10)

    return X_train, X_val, labels, Y_train, Y_val

def look_at_data():
    """
    This function creates a plot of the dataset used to train the model.
    """
    data = complete_dataset()
    data = pd.DataFrame(data)
    data.plot(subplots=True)
    plt.tight_layout()
    plt.show()

def classifier_train(X_train, X_val, Y_train):

    vectoriser = CountVectorizer()
    X_train = vectoriser.fit_transform(X_train, Y_train)
    X_val = vectoriser.transform(X_val)

    pipe = Pipeline([("clf", LogisticRegression(class_weight="balanced"))])

    model = OneVsRestClassifier(pipe)

    model.fit(X_train, Y_train)
    with open("nlp_multilabel/model.pkl", "wb") as file:
        pickle.dump(model, file)
    y_predicted = model.predict(X_val)

    return y_predicted

def classifier_refine(Y_val, y_predicted):

    conf_mat = confusion_matrix(Y_val.argmax(axis=1), y_predicted.argmax(axis=1))
    cm_disp = ConfusionMatrixDisplay(confusion_matrix = conf_mat)
    precision = precision_score(Y_val, y_predicted, average = None)
    recall = recall_score(Y_val, y_predicted, average = None)
    f1 = f1_score(Y_val, y_predicted, average = None)

    cm_disp.plot()
    plt.show()

    print(f"Confusion matrix:\n{conf_mat}\n")
    print(f"Precision:\n{precision}\n")
    print(f"Recall:\n{recall}\n")
    print(f"f1:\n{f1}\n")

    glob_prec = sum(precision) / len(precision)
    glob_rec = sum(recall) / len(recall)
    mac_f1 = 2 *((glob_rec * glob_prec)/(glob_prec + glob_rec))

    print(f"Global precision: {glob_prec}")
    print(f"Global recall: {glob_rec}")
    print(f"Macro f1: {mac_f1}")

if __name__ == "__main__":
   X_train, X_val, labels, Y_train, Y_val = data_prep_main()
   y_predicted = classifier_train(X_train, X_val, Y_train)
   classifier_refine(Y_val, y_predicted)