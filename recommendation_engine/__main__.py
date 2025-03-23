from src.__main__ import grab_deck_list
import json
from nlp_data_gathering.__main__ import manual_supplimentation
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nlp_multilabel.__main__ import preprocess_text

def deck_focus():
    original_deck_list = grab_deck_list()
    deck_list_qantities = []
    for card in original_deck_list:
        deck_list_qantities.append(card[:2])
    for i in range(len(deck_list_qantities)):
        deck_list_qantities[i] = deck_list_qantities[i].rstrip


    deck_arhcetypes = ["voltron", "stax", "group hug", "creatures", "spellslinger", "lands", "graveyard", "artifacts", "counters", "control", "tokens"]
    legends = input("Does your deck care about legendary cards?\n")

def nlp_prep():
    deck_list_formatted, deck_list = manual_supplimentation()
    only_card_txt = []
    for card in deck_list_formatted:
        only_card_txt.append(card[1])
    processed_txt = preprocess_text(only_card_txt)
    return processed_txt, deck_list
    
def nlp_tagging():
    feed_in_text, deck_list = nlp_prep()
    deck_list = [x for x in deck_list if "/" not in x]
    tags = [
        "flicker",
        "proliferate",
        "extra_step/combat", 
        "extra_turn",
        "cast_outside_hand", 
        "draw", 
        "protection",
        "removal", 
        "stat_enhancer", 
        "keyword_granter", 
        "burn", 
        "discard", 
        "recursion", 
        "tokens", 
        "mill", 
        "counterspell", 
        "ramp", 
        "mana_reducers", 
        "copy_spell", 
        "lifegain", 
        "tutor", 
        "counters", 
        "evasion", 
        "stax", 
        "lands_matter", 
        "graveyard_hate", 
        "creature_steal",
        "sacrifice", 
        "untap", 
        "land_destruction",
        "goad",
        "cycling", 
        "tap", 
        "face_down", 
        "graveyard_matters", 
        "modal", 
        "library_filter", 
        "self_buff", 
        "has_keyword"
        ]
    model = pickle.load(open("nlp_multilabel/model.pkl", "rb"))
    predictions = model.predict(feed_in_text)
    
    tags_per_card =[]
    model_tagged_deck = []
    for card in predictions:
        stored_tags = []
        for i, label in enumerate(card):
            if label == 1:
                stored_tags.append("#" + tags[i])
        tags_per_card.append(" " + " ".join(stored_tags))
    
    for i, card in enumerate(deck_list):
        model_tagged_deck.append(card + tags_per_card[i])

    return model_tagged_deck
def typeline_tagging():
    "apply tags based on typeline"
def basic_tagging():
    "apply tags based on presence or absence of specific words ie populate and proliferate"
def focus_tag_filtering():
    "remove irrelevant tags based on info provided in deck focus"












if __name__ == "__main__":
    x = nlp_tagging()
    print(x)
    i = 69