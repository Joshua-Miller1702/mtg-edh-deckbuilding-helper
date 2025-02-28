from src.__main__ import grab_deck_list

def deck_focus():
    "stores dict for user imput"
    "promt users for deck focus (commander vs 99), archetype"
    "also ask if care about legendary permanents ect"
    deck_list = grab_deck_list()
    deck_arhcetypes = ["voltron", "stax", "group hug", "creatures", "spellslinger", "lands", "graveyard", "artifacts", "counters", "control", "tokens"]
    legends = input("Does your deck care about legendary cards?\n")
def nlp_prep():
    deck_list = grab_deck_list()
    
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
    z = 1