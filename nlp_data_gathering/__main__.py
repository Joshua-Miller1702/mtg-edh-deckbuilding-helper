import requests
import json
import csv
from sys import argv
import random
import re
import pandas as pd
from src.__main__ import grab_deck_list

def random_card_grabber(card_count):
    """
    This function grabs some number of random cards from the scryfall database / from local data and returns the card's oracle id, oracle text, name, cmc and type line.

    Args:
        repetitions - The number of random cards to grab.
    """
    #base_url = "https://api.scryfall.com"
    #random_mod_format = base_url + "/cards/random"
    #random_card = requests.get(random_mod_format).json()
    random_cards = []
    keys = ["oracle_id", "oracle_text", "name", "cmc", "type_line", "card_face"]
    with open("data/cards.json", 'r', encoding='utf-8') as cards:
        cards = json.load(cards)
        for card in range(card_count):
            random_card = random.choice(cards)
            values = [random_card.get(key) for key in keys]
            random_cards.append(values)

    return random_cards

def manual_sorting(output_file, deck_list, card_count = None):
    """
    This function allows the user to generate random cards and sort them into buckets then writes it to a CSV.

    Args:
        output_file: The name of the CSV file to write the dataset to.
        card_count: The number of cards to generate and sort.
    """
    cards_dict = {
        "oracle_id": [],
        "oracle_text": []
    }

    card_buckets_dict = {
        "flicker": [],
        "proliferate": [],
        "extra_step/combat": [],
        "extra_turn": [],
        "cast_outside_hand": [],
        "poison": [], #not needed
        "draw": [], 
        "protection": [],
        "removal": [],
        "stat_enhancer": [],
        "keyword_granter": [],
        "burn": [],
        "discard": [],
        "recursion": [],
        "vehicle": [], #not needed
        "tokens": [],
        "mill": [],
        "counterspell": [],
        "ramp": [],
        "mana_reducers": [],
        "alternate_win": [],
        "copy_spell": [],
        "lifegain": [],
        "tutor": [],
        "counters": [],
        "damage_multiplyers": [], #not needed
        "evasion": [],
        "stax": [],
        "lands_matter": [],
        "graveyard_hate": [],
        "creature_steal": [],
        "sacrifice": [],
        "untap": [],
        "land_destruction": [],
        "cheat": [],
        "flexible": [],
        "goad": [], 
        "jank": [],
        "mdfc": [], #not needed
        "cycling": [],
        "tap": [],
        "face_down": [],
        "graveyard_matters": [],
        "modal": [],
        "waste": [],
        "library_filter": [],
        "self_buff": [],
        "has_keyword": []
    }
    if deck_list:
        random_cards = deck_list
        card_count = len(deck_list)
    else:
        random_cards = random_card_grabber(card_count)
        card_count = card_count

    for card in random_cards:
        if card[1] == None:
            card_count -= 1
            continue
        card[1] = card[1].replace(",","")
        card[1] = card[1].replace("\n"," ")
        cards_dict["oracle_id"].append(card[0])
        cards_dict["oracle_text"].append(card[1])
        print(card)
        validation = True
        for key in card_buckets_dict.keys():
            card_buckets_dict[key].append(0)
        while validation:
            try:
                bucket = input("Which bucket does this card belong in: ")
                if bucket == "done":
                    validation = False
                    break
                else:
                    card_buckets_dict[bucket] = card_buckets_dict[bucket][:-1]
                    card_buckets_dict[bucket].append(1)
            except KeyError:
                print("Invalid bucket, please try again.")
                    
    combined_dict = cards_dict | card_buckets_dict
    keys = combined_dict.keys()

    with open(output_file, "w", newline="", encoding = "utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(combined_dict.keys())
        for i in range(card_count):
            writer.writerow(combined_dict[key][i] for key in keys) # MB resolved on line 95? UnicodeEncodeError: 'charmap' codec can't encode character '\u2212' in position 120: character maps to <undefined> - occasionally throws this error
            
    return output_file

def manual_supplimentation(output_file):
    """
    This function allows for manual supplimentation of cards to even out underepresented parts of the dataset. Invokes manual_sorting to label a decklist.

    Args:
        output_file - output_file name (.csv)
        deck_list - not an arg passed in but this function will grab a decklist (or anything) from your clipboard, only in MTGA format.
    """
    deck_list, deck_name = grab_deck_list()
    for i in range(len(deck_list)):
        deck_list[i] = re.sub(r'\d', '', deck_list[i])
        deck_list[i] = deck_list[i].lstrip()

    rdy_to_sort = []
    keys = ["oracle_id", "oracle_text", "name", "cmc", "type_line", "card_face"]
    with open("data/cards.json", 'r', encoding='utf-8') as cards:
        cards = json.load(cards)
        for name in deck_list:
            for card in cards:
                if card["name"] == name:
                    values = [card.get(key) for key in keys]
                    rdy_to_sort.append(values)

    manual_sorting(output_file, deck_list = rdy_to_sort)

if __name__ == "__main__":
    output_file = argv[1]
    """card_count = int(argv[2]) 
    manual_sorting(output_file, card_count)"""
    manual_supplimentation(output_file)
    