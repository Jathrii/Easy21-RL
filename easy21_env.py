from card import Card
from state import State
import numpy as np

# Playable Actions
ACTIONS = (HIT, STICK) = (0, 1)

def draw_card():
        if np.random.rand() > (1/3):
            return np.random.randint(1, 11)
        return -np.random.randint(1, 11)

def bust(sum):
    return sum > 21 or sum < 1

class Easy21Env:

    def reset(self):
        self.state = State(np.random.randint(1,11), np.random.randint(1,11))
        self.reward = 0

    def __init__(self):
        self.reset()

    def step(self, state, action):
        if action == HIT:
            drawn_card = draw_card()
            state.player_sum += drawn_card

            print("New card: " + str(drawn_card))
            print("Sum: " + str(state.player_sum))
            print("Dealer's card: " + str(state.dealer_card))

            if bust(state.player_sum):
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

                if bust(dealer_sum):
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

    def play(self):
        print("Your card: " + str(self.state.player_sum))
        print("Dealer's card: " + str(self.state.dealer_card))

        while self.state is not None:
            print("Hit or Stick?")
            action = eval(input())
            self.state, self.reward = self.step(self.state, action)

        print("Reward: " + str(self.reward))

# game = Easy21Env()
# game.play()