import numpy as np
from card import Card
from state import State

DECK = np.arange(1, 11)
ACTIONS = (HIT, STICK) = (0, 1)
STATE_SHAPE = 2
V_SHAPE = len(DECK), 21
Q_SHAPE = len(DECK), 21, len(ACTIONS)


def draw_card():
    if np.random.rand() > (1/3):
        return np.random.choice(DECK)
    return -np.random.choice(DECK)


def bust(sum):
    return sum > 21 or sum < 1


class Easy21Env:

    def reset(self):
        self.state = State(np.random.choice(DECK), np.random.choice(DECK))
        self.reward = 0
        return self.state

    def __init__(self, display=True):
        self.display = display
        self.reset()

    def step(self, state, action):
        if action == HIT:
            drawn_card = draw_card()
            state.player_sum += drawn_card
            
            if self.display:
                print("New card: " + str(drawn_card))
                print("Sum: " + str(state.player_sum))
                print("Dealer's card: " + str(state.dealer_card))

            if bust(state.player_sum):
                if self.display:
                    print("You went bust! You lose!")
                return None, -1

            return state, 0
        elif action == STICK:
            dealer_sum = state.dealer_card

            while dealer_sum < 17:
                drawn_card = draw_card()
                dealer_sum += drawn_card

                if self.display:
                    print("Dealer's new card: " + str(drawn_card))
                    print("Dealer's sum: " + str(dealer_sum))

                if bust(dealer_sum):
                    if self.display:
                        print("Dealer went bust! You win!")
                    return None, 1

            if self.display:
                print("Dealer sticks!")

            if state.player_sum > dealer_sum:
                if self.display:
                    print("You win!")
                return None, 1
            elif state.player_sum < dealer_sum:
                if self.display:
                    print("You Lose!")
                return None, -1
            else:
                if self.display:
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
