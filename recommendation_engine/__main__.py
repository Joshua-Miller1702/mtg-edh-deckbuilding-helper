from src.__main__ import grab_deck_list
import json
from nlp_data_gathering.__main__ import manual_supplimentation
import re
from nlp_multilabel.__main__ import preprocess_text

def deck_focus():
    "stores dict for user imput"
    "promt users for deck focus (commander vs 99), archetype"
    "also ask if care about legendary permanents ect"
    deck_list = grab_deck_list()
    deck_arhcetypes = ["voltron", "stax", "group hug", "creatures", "spellslinger", "lands", "graveyard", "artifacts", "counters", "control", "tokens"]
    legends = input("Does your deck care about legendary cards?\n")

def nlp_prep():
    deck_list_formatted = manual_supplimentation()
    only_card_txt = []
    for card in deck_list_formatted:
        only_card_txt.append(card[1])
    processed_txt = preprocess_text(only_card_txt)
    return processed_txt
    
def nlp_tagging():
    "modify decklist so it can be input to nlp model"
    "run decklist through the NLP model to tag it with all those"
def typeline_tagging():
    "apply tags based on typeline"
def basic_tagging():
    "apply tags based on presence or absence of specific words ie populate and proliferate"
def focus_tag_filtering():
    "remove irrelevant tags based on info provided in deck focus"












if __name__ == "__main__":
    x= nlp_prep()
    print(x)