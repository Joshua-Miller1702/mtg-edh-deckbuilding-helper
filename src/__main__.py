import sys
import math

#Potential expansions listed below
#Deck speed variable to help with below thresholds
#Deck focus ----> cardtype threshold recomendations 95% chance to draw in first 5 turns so first 12 cards, mb modified for draw cards.
#Accept user imput in args rather than manually changing variables
#Frontend app/ui/interface, simple but useful

def hypergeo_calculator(total_desired_cards = int, number_in_hand = int, cards_drawn = 7, deck_size = 99):
    """ 
    This calcuclates the percentage chance that a specific type of card will be drawn from a deck of a given size.

    Args:
        total_desired_cards: The amount of a desired card type in the deck
        cards_drawn: The total number of cards drawn
        number_in_hand: The number of a desired card type wanted in hand
        deck_size: The total number of cards in deck
    """
    try:
        success_in_pop = ((math.comb(total_desired_cards, number_in_hand))*(math.comb(deck_size - total_desired_cards, cards_drawn - number_in_hand))) / (math.comb(deck_size, cards_drawn))
        percentage_chance = round(success_in_pop*100, 2)
        return percentage_chance
    except ValueError:
        print("Your numbers are not compatible!")
        

if __name__ == "__main__" :
    
    total_desired_cards = int(input("How many cards of the desired type are in your deck: "))
    number_in_hand = int(input("How many would you like in hand: "))
    cards_drawn = int(input("How many cards will you draw: "))
    deck_size = int(input("How many cards are in your deck: "))

    percentage_chance = hypergeo_calculator(total_desired_cards, number_in_hand, cards_drawn, deck_size)
    print(f"You will hit this draw {percentage_chance:00.2f}% of the time!")
    
