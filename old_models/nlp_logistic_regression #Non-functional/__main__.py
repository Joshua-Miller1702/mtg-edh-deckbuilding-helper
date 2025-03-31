import re
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import nltk
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.feature_extraction.text import CountVectorizer
import json


def retrieve_card_text_train(card_ids: list = None):
    """
    This function retrieves the oracle text from the scryfall API for a list of card ids in the training set.

    Args: 
        card_ids: A list of card ids to retrieve the oracle text for.
    """
    base_url = "https://api.scryfall.com/cards/"
    card_ids = card_ids if card_ids else []

    with open("data/train_validate.csv", "r", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter = ",")
        for row in reader:
            card_ids.append(row[0])
    card_ids.pop(0)
    cards_text = []
    with open("data/cards.json", 'r', encoding='utf-8') as cards:
        cards_data = json.load(cards)
        for card_id in card_ids:
            card_data = None
            for card in cards_data:
                if card['uri'][31:] == card_id:
                    card_data = card
                    break
            key = "oracle_text"
            try:
                cards_text.append(card_data[key])
            except:
                cards_text.append('')

    
    cards_text = [str(i) for i in cards_text]

    return cards_text

def preprocess_text(cards_text: list = None):
    """
    This function preprocesses the card text in perpartion for NLP training.

    Args:
        cards_text: A list of cards text to process.
    """
    cards_text = cards_text if cards_text else retrieve_card_text_train()

    for i in range(len(cards_text)):
        cards_text[i] = re.sub(r'\d+', '', cards_text[i])
        cards_text[i] = re.sub(r'[^\w\s]', '', cards_text[i])
        cards_text[i] = re.sub(r'\n', ' ', cards_text[i])
        cards_text[i] = cards_text[i].lower()

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
    data = data.loc[data['text'] != '']
    text = (data["text"].values)
    labels = torch.tensor(data.loc[:, data.columns != "text"].values)
    
    train_text, test_text, train_labels, test_labels = train_test_split(text, labels, test_size=0.25, random_state=17)

    return data, train_text, test_text, train_labels, test_labels

def get_vocab(train_text, test_text, train_labels, test_labels):
    """
    This function generates a vocabulary of unique words for use in the NLP model using bag of words.

    Args:
        train_text: the list of sentences used to generate the vocab dictionary.
        test_text: text to be tansformed to use when testing the model.
    Returns:
        vocab: dictionary of unique words and an index that can be used as a tensor
    """
    vectorizer = CountVectorizer()
    trans_train_text = vectorizer.fit_transform(train_text)
    trans_test_text = vectorizer.transform(test_text)
    tens_train_text = torch.from_numpy(trans_train_text.todense())
    tens_test_text = torch.from_numpy(trans_test_text.todense())
    train_dataset = TensorDataset(tens_train_text, train_labels)
    test_dataset = TensorDataset(tens_test_text, test_labels)
    return tens_train_text, tens_test_text, train_dataset, test_dataset

class Logistic_regression_model(torch.nn.Module):
    def __init__(self, input_feat_count, num_classes):
        super(Logistic_regression_model, self).__init__()
        self.linear = nn.Linear(input_feat_count, num_classes)
    
    def forward(self, x):
        """
        Applies softmax function 
        """
        x = x.to(torch.float32)
        out = self.linear(x)
        # out = nn.functional.softmax(out, dim = 1)
        return out


if __name__ == "__main__":
    #get data
    data, train_text, test_text, train_labels, test_labels = final_data_prep(train_test = "data/train_validate.csv")
    
    tens_train_text, tens_test_text, train_dataset, test_dataset = get_vocab(train_text, test_text, train_labels, test_labels)
    num_classes = train_labels.shape[1]
    input_feat_count = tens_train_text.shape[1]
    #batches
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
    #model setup
    model = Logistic_regression_model(input_feat_count, num_classes = 35)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, input_feat_count = (batch_size, input_feat_count))
    #loss and grad desc set up
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    # loss_func = nn.MSELoss()
    grad_desc = torch.optim.SGD(model.parameters())
    # grad_desc = torch.optim.Adagrad(model.parameters(), lr = 0.1)
    epochs = 1
    #train
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(torch.float32)
            labels = labels.to(device)

            grad_desc.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            
            loss.backward()
            grad_desc.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    #evaluate
    with torch.no_grad():
        correct = 0
        total = 0
        true_positives = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            outputs = model(inputs)
            threshold = 1.0
            outputs = (outputs > threshold).float()
            # _, predicted = torch.max(outputs.data, 1)

            total += (labels.shape[0]*labels.shape[1])
            
            response = torch.stack((labels, outputs), axis=2)
            """true_positives += (response =)"""
            correct += (outputs == labels).sum().item()

        print(f"Accuracy: {(100 * correct / total):.2f}%")

