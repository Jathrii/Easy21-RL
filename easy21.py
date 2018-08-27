from card import Card
from state import State
import numpy as np

# Playable Actions
HIT = 0
STICK = 1

def draw_card():
    if np.random.rand() > (1/3):
        return np.random.randint(1,11)
    return -np.random.randint(1,11)


def step(state, action):
    if action == HIT:
        drawn_card = draw_card()
        state.player_sum += drawn_card

        print("New card: " + str(drawn_card))
        print("Sum: " + str(state.player_sum))
        print("Dealer's card: " + str(state.dealer_card))

        if state.player_sum > 21 or state.player_sum < 1:
            print("You went bust! You lose!")
            return None, -1

        return state, 0
    elif action == STICK:
        dealer_sum = state.dealer_card

        while dealer_sum < 17:
            drawn_card = draw_card()
            dealer_sum += drawn_card

            print("Dealer's new card: " + str(drawn_card))
            print("Dealer's sum: " + str(dealer_sum ))

            if dealer_sum > 21 or dealer_sum < 1:
                print("Dealer went bust! You win!")
                return None, 1

        print("Dealer sticks!")

        if state.player_sum > dealer_sum:
            print("You win!")
            return None, 1
        elif state.player_sum < dealer_sum:
            print("You Lose!")
            return None, -1
        else:
            print("It's a draw!")
            return None, 0

def play():
    dealer_card = np.random.randint(1,11)
    player_card = np.random.randint(1,11)
    print("Your card: " + str(player_card))
    print("Dealer's card: " + str(dealer_card))

    state = State(dealer_card, player_card)
    reward = 0

    while state is not None:
        print("Hit or Stick?")
        action = eval(input())
        state, reward = step(state, action)

    print("Reward: " + str(reward))