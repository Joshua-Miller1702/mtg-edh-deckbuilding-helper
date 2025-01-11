import requests
import json

#name, cmc, colour identity, type_line, text, legalities, power, toughness
def random_card_grabber():
    base_url = "https://api.scryfall.com"
    random_mod_format = base_url + "/cards/random"
    random_card = requests.get(random_mod_format).json()
    keys = ["id", "name", "cmc", "color_identity", "type_line", "oracle_text", "power", "toughness", "legalities"]
    random_card = [random_card.get(key) for key in keys]
    return random_card

if __name__ == "__main__":
    print(random_card_grabber()[-1]['commander'])





#grab ~100 cards from the magic api

#Make card function buckets

#also make buckets for card types

#sort cards into the buckets manually

#make a csv file with the card names and the buckets of half the cards ~50% cards
#make a second file for validation with ~5% of cards
#Make a test file with 45% of cards to test the model

#implement some kind of nlp model (naieve bayesian) to autosort the cards into the buckets

#test using test file to see if the model is accurate

#hope to god that the model is pretty good the first time round :)