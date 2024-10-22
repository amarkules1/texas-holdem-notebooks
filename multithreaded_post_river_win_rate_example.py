from texas_hold_em_utils.deck import Deck
from texas_hold_em_utils.relative_ranking import rank_hand_post_river
import pandas as pd
from multiprocessing import Pool
import datetime

def generate_random_n_player_win_rate(i):
    deck = Deck()
    deck.shuffle()
    two_p_win_rate = rank_hand_post_river([deck.draw(), deck.draw()], [deck.draw(), deck.draw(), deck.draw(), deck.draw(), deck.draw()])
    return two_p_win_rate ** (i-1)



if __name__ == '__main__':
    data = []
    print(f'start time: {datetime.datetime.now()}')
    
    for i in [3,4,5,6,7,8,9,10,11,12]:
        print(f'{i} player sample start time: {datetime.datetime.now()}')
        data = []
        with Pool(16) as p:
            data = p.map(generate_random_n_player_win_rate, [i for j in range(25000)])
        df = pd.DataFrame(data={'player_ct': [i for j in range(25000)], 'win_rate': data})
        df.to_csv(f'data/{i}_player_post_river_win_rates.csv', index=False)
    
    print(f'end time: {datetime.datetime.now()}')
    