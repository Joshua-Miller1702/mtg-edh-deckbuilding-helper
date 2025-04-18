from src.__main__ import grab_deck_list
import json
from nlp_data_gathering.__main__ import manual_supplimentation
import pickle
from nlp_multilabel.__main__ import preprocess_text
import re

def deck_quant():
    original_deck_list, name = grab_deck_list()
    card_quantities = [card[0:2] for card in original_deck_list]

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

def legend_tagging():
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
    
    legend = []
    for line in type_lines:
        if "legendary" in line:
            legend.append("Legendary")
        else:
            legend.append("")
    return legend



def desired_tags():
    tags = ["flicker","proliferate","extra_step/combat", "extra_turn","cast_outside_hand", "draw", "protection","removal", "stat_enhancer", "keyword_granter", "burn", "discard", "recursion", "tokens", "mill", "counterspell", "ramp", "mana_reducers", "copy_spell", "lifegain", "tutor", "counters", "evasion", "stax", "lands_matter", "graveyard_hate", "creature_steal","sacrifice", "untap", "land_destruction","goad","cycling", "tap", "face_down", "graveyard_matters", "modal", "library_filter", "self_buff", "has_keyword"]
    print("Which tags would you like to be applied from the list below (draw, removal and ramp are defaults)\n"
        "Flicker,\n"
        "Proliferate,\n"
        "Extra_step/combat,\n"
        "Extra_turn,\n"
        "Cast_outside_hand,\n"
        "Protection,\n"
        "Stat_enhancer,\n"
        "Keyword_granter,\n"
        "Burn,\n"
        "Discard,\n"
        "Recursion,\n"
        "Tokens,\n"
        "Mill,\n"
        "Counterspell,\n"
        "Mana_reducers,\n"
        "Copy_spell,\n"
        "Lifegain,\n"
        "Tutor,\n"
        "Counters,\n"
        "Evasion,\n"
        "Stax,\n"
        "Lands_matter,\n"
        "Graveyard_hate,\n"
        "Creature_steal,\n"
        "Sacrifice,\n"
        "Untap,\n"
        "Land_destruction,\n"
        "Goad,\n"
        "Cycling,\n"
        "Tap,\n"
        "Face_down,\n"
        "Graveyard_matters,\n"
        "Modal,\n"
        "Library_filter,\n"
        "Self_buff,\n" 
        "Has_keyword.\n" 
        "Please enter desired tags as a list separated by comas")
    final_tags = []
    while True:
        try:  
            desired_tags = input()
            for i in desired_tags:
                if i.isdigit():
                    raise ValueError
            separate_tags = desired_tags.split(", ")
            for i in separate_tags:
                if i not in tags:
                    raise ValueError
            final_tags = separate_tags
            break
        except ValueError:
            print("Invalid input try again")
            continue


    print("Would you like legend tags (y/n)?")
    while True:
        try:
            legend_tag_q = input()
            if not legend_tag_q.isalpha():
                raise ValueError
            legend_tag_q = legend_tag_q.lower()
            if legend_tag_q == "y":
                legend_tag_bool = True
            elif legend_tag_q == "n":
                legend_tag_bool = False
            else:
                raise ValueError
            break
        except ValueError:
            print("Invalid input try again")
            continue


    return final_tags, legend_tag_bool

def focus_tag_filtering():
    user_tags, legend_tag_bool = desired_tags()
    legendary = legend_tagging()
    nlp_tagged_deck = nlp_tagging()
    card_quantities = deck_quant()

    # - filter by desired tags
    for i in range(len(nlp_tagged_deck)):
        for word in nlp_tagged_deck[i].split(" "):
            if word == "":
                continue
            if ((word[0] == "#") and (word not in user_tags)):
                word.replace(word, "")

    # - add legend tags
    if legend_tag_bool:
        for i in range(len(legendary)):
            legendary[i] = "¬" + legendary[i] 
        legend_tagged = [a + b for a, b in zip(nlp_tagged_deck, legendary)]
        legend_tagged = ",¬".join(legend_tagged)
    else:
        legend_tagged = nlp_tagged_deck

    # - add numbers back
    for i in range(len(card_quantities)):
        card_quantities[i] = card_quantities[i] + "¬"
    final_list = [a + b for a, b in zip(card_quantities, nlp_tagged_deck)]
    final_list = "¬,".join(final_list)

    for i in range(len(final_list)):
        final_list[i] = re.sub("¬", "", str(final_list[i]))

    return final_list












if __name__ == "__main__":
    i = focus_tag_filtering()
    print(i)
    x = 2