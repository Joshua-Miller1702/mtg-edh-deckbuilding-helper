import sys
import math
import os
import win32clipboard
import re

#Potential expansions listed below
#Deck speed variable to help with below thresholds
#Deck focus ----> cardtype threshold recomendations 95% chance to draw in first 5 turns so first 12 cards, mb modified for draw cards.
#Accept user imput in args rather than manually changing variables
#Frontend app/ui/interface, simple but useful

def hypergeo_calculator(total_desired_cards: int, number_in_hand: int, cards_drawn = 7, deck_size = 99):
    """ 
    This calcuclates the percentage chance that a specific type of card will be drawn from a deck of a given size.

    Args:
        total_desired_cards: The amount of a desired card type in the deck
        cards_drawn: The total number of cards drawn
        number_in_hand: The number of a desired card type wanted in hand
        deck_size: The total number of cards in deck
    """
    try:
        success_in_pop = []
        counter = number_in_hand
        less_than_desired = 0
        for i in range(number_in_hand + 1):
            calculation = ((math.comb(total_desired_cards, counter))*(math.comb(deck_size - total_desired_cards, cards_drawn - counter))) / (math.comb(deck_size, cards_drawn))
            success_in_pop.append(calculation)
            counter -= 1
        for i in success_in_pop[1:]:
            less_than_desired += i
        percentage_chance_exact = round(success_in_pop[0]*100, 2)
        percentage_chance_more = round((1 - less_than_desired)*100, 2)
        return percentage_chance_exact, percentage_chance_more
    except ValueError:
        print("Your numbers are not compatible, please re-enter!")

def inc_num(card):
    return any(i.isdigit() for i in card)

def grab_deck_list():
    """
    This function will grab the deck list from the users clipboard and return the deck list as an array containing the card quantity and name, also returns the deck name.
    """
    ###!!!!!!Needs to be widened to support other deck import formats, currently only supports MTGA decklists!!!!!!###
    win32clipboard.OpenClipboard()
    deck_list = win32clipboard.GetClipboardData()
    deck_list_arr = deck_list.split("\n")
    deck_list_formatted = []

    deck_name = deck_list_arr[1]
    deck_name_formatted = deck_name.replace("Name ", "")
    
    for i, card in enumerate(deck_list_arr):
        if inc_num(card):
            deck_list_formatted.append(card.strip())
    win32clipboard.CloseClipboard()
    return deck_list_formatted, deck_name_formatted
        

if __name__ == "__main__" :
    
    total_desired_cards = int(input("How many cards of the desired type are in your deck: "))
    number_in_hand = int(input("How many would you like in hand: "))
    cards_drawn = int(input("How many cards will you draw: "))
    deck_size = int(input("How many cards are in your deck: "))

    percentage_chance_exact, percentage_chance_more = hypergeo_calculator(total_desired_cards, number_in_hand, cards_drawn, deck_size)
    print(f"You will draw exactly {number_in_hand} cards {percentage_chance_exact:00.2f}% of the time!")
    print(f"You will draw {number_in_hand} cards or more {percentage_chance_more:00.2f}% of the time!")

    """ deck_list, deck_name = grab_deck_list()
    print(deck_list, deck_name) """
    
