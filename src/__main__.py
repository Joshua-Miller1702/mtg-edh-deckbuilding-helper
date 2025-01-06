import sys
import math


if __name__ == "__main__" :
    
    start = sys.argv[1]

    deck_size, N = 100, 100
    desired_card_count, K = 14, 14
    cards_drawn, n = 7, 7
    number_in_hand, k = 2, 2

    success_in_pop = ((math.comb(desired_card_count, k))*(math.comb(deck_size - desired_card_count, cards_drawn - number_in_hand))) / (math.comb(deck_size, cards_drawn))
    percentage = round(success_in_pop*100, 2)
    print(str(percentage) + "%")


    