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

def manual_sorting(output_file, deck_list, card_count = 0):
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
        "flicker": [], #Needed
        "proliferate": [], #Not needed can use hard rule "Proliferate"
        "extra_step/combat": [], #Needed
        "extra_turn": [], #Not needed can use hard rule "extra turn"
        "cast_outside_hand": [], #Needed
        "poison": [], #not needed (not useful)
        "draw": [], #Needed
        "protection": [], #Needed
        "removal": [], #Needed
        "stat_enhancer": [], #Needed
        "keyword_granter": [], #Needed
        "burn": [], #Needed
        "discard": [], #Not needed can use hard rule "discard"
        "recursion": [], #Needed
        "vehicle": [], #not needed (not useful)
        "tokens": [], #Needed
        "mill": [], #Needed
        "counterspell": [], #Needed
        "ramp": [], #Needed
        "mana_reducers": [], #Needed
        "alternate_win": [], #Not needed (not useful)
        "copy_spell": [], #Needed
        "lifegain": [], #Needed
        "tutor": [], #Needed
        "counters": [], #Needed
        "damage_multiplyers": [], #not needed (too few)
        "evasion": [], #Needed
        "stax": [], #Needed
        "lands_matter": [], #Needed
        "graveyard_hate": [], #Needed
        "creature_steal": [], #Not needed, "Gain control of target creature"
        "sacrifice": [], #Needed
        "untap": [], #Not needed (dont use often)
        "land_destruction": [], #Needed
        "cheat": [], #Not needed (not useful)
        "flexible": [], #Not needed (not useful)
        "goad": [], #Can keep
        "jank": [], #Not needed (not useful)
        "mdfc": [], #not needed (dosent work)
        "cycling": [], #Needed
        "tap": [], #Needed
        "face_down": [], #Needed
        "graveyard_matters": [], #Needed
        "modal": [], #Needed
        "waste": [], #tag to put when mistakes are made, samples are dropped.
        "library_filter": [], #Needed
        "self_buff": [], #Needed
        "has_keyword": [] #Needed
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
            writer.writerow(combined_dict[key][i] for key in keys) 
            
    return output_file

def manual_supplimentation():
    """
    This function allows for manual supplimentation of cards to even out underepresented parts of the dataset. Invokes manual_sorting to label a decklist.
    """
    deck_list, deck_name = grab_deck_list()
    for i in range(len(deck_list)):
        deck_list[i] = re.sub(r'\d', '', str(deck_list[i]))
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
    
    return rdy_to_sort , deck_list

if __name__ == "__main__":
    output_file = argv[1]
    """card_count = int(argv[2]) 
    manual_sorting(output_file, card_count)"""
    deck_list_formatted, deck_list = manual_supplimentation()
    manual_sorting(output_file, deck_list_formatted)
    
