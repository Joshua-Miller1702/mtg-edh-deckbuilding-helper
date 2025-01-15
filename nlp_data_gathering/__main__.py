import requests
import json
import csv
from sys import argv


def random_card_grabber(repetitions = 1):
    """
    This function grabs some number of random cards from the scryfall database and returns the card's id and oracle text.

    Args:
        repetitions: The number of random cards to grab from the scryfall database.
    """
    base_url = "https://api.scryfall.com"
    random_mod_format = base_url + "/cards/random"
    random_card = requests.get(random_mod_format).json()
    keys = ["id", "oracle_text", "name", "cmc", "color_identity", "type_line", "power", "toughness"]
    random_card = [random_card.get(key) for key in keys]
    return random_card

def manual_sorting(output_file, card_count = 100):
    """
    This function allows the user to generate random cards and sort them into buckets then writes it to a CSV.

    Args:
        output_file: The name of the CSV file to write the dataset to.
        card_count: The number of cards to generate and sort.
    """
    cards_dict = {
        "id": [],
    }

    card_buckets_dict = {
        "land": [],
        "creature": [],
        "artifact": [],
        "enchantment": [],
        "planeswalker": [],
        "battle": [],
        "instant": [],
        "sorcery": [],
        "kindred": [],
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
        "graveyard_hate": [],
        "creature_steal": [],
    }

    for i in range(card_count):
        card = random_card_grabber()
        cards_dict["id"].append(card[0])
        print(card)
        validaiton = True
        for key in card_buckets_dict.keys():
            card_buckets_dict[key].append(0)
        while validaiton:
            try:
                bucket = input("Which bucket does this card belong in: ")
                if bucket == "done":
                    validaiton = False
                    break
                else:
                    card_buckets_dict[bucket] = card_buckets_dict[bucket][:-1]
                    card_buckets_dict[bucket].append(1)
            except KeyError:
                print("Invalid bucket, please try again.")
                    
            

    combined_dict = cards_dict | card_buckets_dict
    keys = combined_dict.keys()

    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(combined_dict.keys())
        for i in range(card_count):
            writer.writerow(combined_dict[key][i] for key in keys)
            
    return output_file

#make a csv file with the card names and the buckets of half the cards ~50% cards
#make a second file for validation with ~5% of cards
#Make a test file with 45% of cards to test the model

#implement some kind of nlp model (naieve bayesian) to autosort the cards into the buckets

#test using test file to see if the model is accurate

#hope to god that the model is pretty good the first time round :)



if __name__ == "__main__":
    output_file = argv[1]
    card_count = int(argv[2])
    manual_sorting(output_file, card_count)