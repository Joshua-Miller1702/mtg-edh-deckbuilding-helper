from src.__main__ import grab_deck_list
import json
from nlp_data_gathering.__main__ import manual_supplimentation
import pickle
from nlp_multilabel.__main__ import preprocess_text

def deck_focus():
    original_deck_list, name = grab_deck_list()
    card_quantities = [card[0:2] for card in original_deck_list]
    
    """deck_arhcetypes = ["voltron", "stax", "group hug", "creatures", "spellslinger", "lands", "graveyard", "artifacts", "counters", "control", "tokens"]
    legends = input("Does your deck care about legendary cards?\n")"""

    return card_quantities

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
    text, deck_list = nlp_prep()

    type_lines = []
    with open("data/cards.json", 'r', encoding='utf-8') as cards:
        cards = json.load(cards)
        for name in deck_list:
            x = ""
            for card in cards:
                if card["name"] == name:
                    x = card.get("type_line")
                    type_lines.append(x)
    for i in range(len(type_lines)):
        type_lines[i] = type_lines[i].lower()
    return type_lines


def focus_tag_filtering():
    type_lines = typeline_tagging()
    nlp_tags = nlp_tagging()
    card_quantities = deck_focus()
    "adds card numbers back to deck"
    "remove irrelevant tags based on info provided in deck focus"












if __name__ == "__main__":
    i = deck_focus()
    print(i)
    x = 1