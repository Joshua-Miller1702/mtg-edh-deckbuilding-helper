import pandas as pd
import json
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk.corpus
import itertools
import csv
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.naive_bayes import MultinomialNB

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
    data.insert(loc = 0, column = "processed_text", value = processed_text)

    return data

def data_prep(data: csv = None): #May need to change this to split up label values so i can use them as a list to iterate over so the model can calculate each probablity separately
    #possibly use dataloader batches = 1 to load each thing on its own idk

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

    label_values = torch.tensor(data.loc[:, data.columns != "processed_text"].values)
    text_data = data["processed_text"].values
    labels = data.columns
    labels = labels[1:]

    X_train, X_validate, Y_train, Y_validate = train_test_split(text_data, label_values, test_size = 0.25, random_state = 17)

    vectorizer = CountVectorizer()
    X_train_fit = vectorizer.fit_transform(X_train)
    X_validate_fit = vectorizer.transform(X_validate) #fit to vectorizer not model?

    X_train_tens = torch.from_numpy(X_train_fit.todense())
    X_validate_tens = torch.from_numpy(X_validate_fit.todense())
    train_dataset = TensorDataset(X_train_tens, Y_train)
    validate_dataset = TensorDataset(X_validate_tens, Y_validate)

    return train_dataset, validate_dataset

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

class NaieveBayes:

    def get_prior_mean_var(self, X):
        #Prior = count how often class label occours
        #P(xi|y) = gausian model, need mean and var
        n_samples = len(X)
        n_classes = len(X[0][1])
        #======================================================= get prior =================================================================#
        #Calculate for each class not each sample
        class_frequency = np.zeros(n_classes, dtype=float)
        for i in range(n_classes):
            count = 0.0
            for j in range(n_samples):
                count += (X[j][1][i])
                if count > 0:
                    class_frequency[i] = count

        self._priors = []
        for i in class_frequency:
            if i > 0:
                self._priors.append(i / n_samples)
            else:
                self._priors.append(0.0)

        #====================================================== get mean ===================================================================#
        #mean of all features in given class 
        self._means = []
        for i in range(n_classes):
            total = 0.0
            count = 0.0
            for j in range(n_samples):
                if X[0][1][i] == 1:
                    count += 1
                    total += sum(X[j][0])
            try:
                self._means.append(total / count * n_samples)
            except ZeroDivisionError:
                self._means.append(0)    

        #====================================================== get var =====================================================================#
        #variance of all features in a given class
        self._vars = []
        for i in range(n_classes):
            arrays = []
            for j in range(n_samples):
                if X[0][1][i] == 1:
                    arrays.append(X[j][0])
            arrays = list(itertools.chain(*arrays))
            self._vars.append(np.var(arrays))


    def _pdf(self, class_i, x):

        mean = self._means[class_i]
        var = self._vars[class_i]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return (numerator/denominator)


    def predict(self, X):
        class_pred = [self._predict(X, x) for x in X]
        return class_pred


    def _predict(self, X, x):
        #calculate posteriors for each class
        n_classes = len(X[0][1])

        posteriors = []
        
        for i in range(n_classes):
            prior = np.log(self._priors[i])
            posterior = np.sum(np.log(self._pdf(i, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        
        x_classes = np.zeros(n_classes)
        threshold = 0.7
        
        #return classes with posteriors above a threshold
        for i, posterior in enumerate(posteriors):
            if posterior[i] > threshold:
                x_classes[i] = 1

        return x_classes.nonzero()



if __name__ == "__main__":
   """ X, Val = data_prep()

    def accuracy(lab_true, lab_pred):
        accuracy = np.sum(lab_true == lab_pred) / len(lab_true)
        return accuracy
    
    nb = NaieveBayes()
    nb.get_prior_mean_var(X)
    predictions = nb.predict(Val)
    
    print(f"Accuracy: {accuracy(Val, predictions)}")"""
   
   X, Valx, Y, Valy = data_prep_alt()
   
   model = MultinomialNB()
   model.fit(X, Y)
   print(model.predict(Valx))
