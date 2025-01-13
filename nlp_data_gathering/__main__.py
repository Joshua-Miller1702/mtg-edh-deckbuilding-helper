import requests
import json
import csv

"""
"id", "name", "cmc", "color_identity", "type_line", "oracle_text", "power", "toughness" - will want to pull all these later for final stuff but not while making the dataset.
"""

def random_card_grabber(repetitions = 1):
    """
    This function grabs some number of random cards from the scryfall database and returns the card's id and oracle text.

    Args:
        repetitions: The number of random cards to grab from the scryfall database.
    """
    base_url = "https://api.scryfall.com"
    random_mod_format = base_url + "/cards/random"
    random_card = requests.get(random_mod_format).json()
    keys = ["id", "oracle_text"]
    random_card = [random_card.get(key) for key in keys]
    return random_card

"""
The below code is for manually sorting cards into buckets and writing it to a CSV for training an NLP model.
"""
def manual_sorting(card_count = 100):
    cards_dict = {
        "id": [],
        "oracle_text": []
    }

    card_buckets_dict = {
        "draw": [], 
        "protection": [],
        "removal": [],
        "stat_enhancer": [],
        "keyword_granter": [],
        "burn": [],
        "discard": [],
        "recursion": [],
        "tokens": [],
        "mill": [],
        "counterspell": [],
        "ramp_and_mana_reducers": [],
        "alternate_win": [],
        "copy_spell": [],
        "lifegain": [],
        "tutor": [],
        "counters": [],
        "damage_multiplyers": [],
        "evasion": [],
        "stax": [],
        "lands_matter": [],
        "graveyard_hate": []
    }

    for i in range(card_count):
        card = random_card_grabber()
        cards_dict["id"].append(card[0])
        cards_dict["oracle_text"].append(card[1])
        print(card)
        validaiton = True
        while validaiton:
            bucket = input("Which bucket does this card belong in: ")
            if bucket == "done":
                validaiton = False
                break
            else:
                card_buckets_dict[bucket].append(1)
        for key in card_buckets_dict.keys():
            if key[i] != 1:
                card_buckets_dict[key].append(0)



#sort cards into the buckets manually

#make a csv file with the card names and the buckets of half the cards ~50% cards
#make a second file for validation with ~5% of cards
#Make a test file with 45% of cards to test the model

#implement some kind of nlp model (naieve bayesian) to autosort the cards into the buckets

#test using test file to see if the model is accurate

#hope to god that the model is pretty good the first time round :)



if __name__ == "__main__":
    manual_sorting()