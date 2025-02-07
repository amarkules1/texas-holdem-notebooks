from texas_hold_em_utils.deck import Deck
from texas_hold_em_utils.game import Game
from texas_hold_em_utils.player import KellyMaxProportionPlayer
import pandas as pd
from multiprocessing import Pool
import datetime
import math
import random

def simulate_n_player_round(n):
    game = Game(num_players=n)
    kelly_props = [math.ceil(random.random() * 20) / 20.0 for _ in range(n)]
    players = [KellyMaxProportionPlayer(i, round_proportions=[kelly_props[i],kelly_props[i],kelly_props[i],kelly_props[i]]) for i in range(n)]
    game.players = players
    game.run_round()
    return pd.DataFrame(data={'kelly_proportion': kelly_props, 'stack': [player.chips for player in players]})


if __name__ == '__main__':
    data = []
    print(f'start time: {datetime.datetime.now()}')
    
    for i in [6,7,8,9,10]:
        print(f'{i} player sample start time: {datetime.datetime.now()}')
        data = []
        with Pool(14) as p:
            data = p.map(simulate_n_player_round, [i for j in range(8000)])
        df = pd.concat(data)
        df.to_csv(f'data/{i}_player_kelly_proportions.csv', index=False)
    
    print(f'end time: {datetime.datetime.now()}')