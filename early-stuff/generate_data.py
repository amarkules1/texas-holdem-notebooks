import texas_hold_em_utils.card as card
import texas_hold_em_utils.deck as deck
import texas_hold_em_utils.game as game
import texas_hold_em_utils.player as player
import pandas as pd

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import os
import sqlalchemy


conn = sqlalchemy.create_engine(os.getenv("SUPABASE_CONN_STRING")).connect()

def simulate_random_game(num_players):
    game1 = game.Game(num_players)
    game1.players = [player.SimplePlayer(i) for i in range(num_players)]
    game1.deal()
    game1.get_bets()
    game1.flop()
    game1.get_bets()
    game1.turn()
    game1.get_bets()
    game1.river()
    game1.get_bets()
    winners = game1.determine_round_winners()
    df = pd.DataFrame(data={
        "flop_1": [game1.community_cards[0].__str__()],
        "flop_2": [game1.community_cards[1].__str__()],
        "flop_3": [game1.community_cards[2].__str__()],
        "turn": [game1.community_cards[3].__str__()],
        "river": [game1.community_cards[4].__str__()],
        "winners": ", ".join([f"player_{winner.position}" for winner in winners])
    })
    for i, player1 in enumerate(game1.players):
        df[f"player_{i}_card_1"] = player1.hand_of_two.cards[0].__str__()
        df[f"player_{i}_card_2"] = player1.hand_of_two.cards[1].__str__()
        df[f"player_{i}_hand_rank"] = player1.hand_of_five.hand_rank
    return df

for num_players in range(9,13):
    for i in range(1000 // (num_players - 1)):
        data_df = None
        for j in range(1000):
            if data_df is None:
                data_df = simulate_random_game(num_players)
            else:
                data_df = pd.concat([data_df, simulate_random_game(num_players)])
        data_df.to_sql(f"random_games_for_{num_players}_players", conn, if_exists="append", schema="poker")
        conn.commit()
        
conn.commit()
conn.close()