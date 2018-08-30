class State:

    def __init__(self, dealer_card, player_sum):
        self.dealer_card = dealer_card
        self.player_sum = player_sum

    def to_arr(self):
        return [self.dealer_card, self.player_sum]
