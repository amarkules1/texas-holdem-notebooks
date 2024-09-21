import pandas as pd

def is_start_time_line(line):
    return line.startswith('Game started at: ')

def is_game_id_line(line):
    return line.startswith('Game ID: ')

def get_game_id(line):
    return line.split(' ')[2]

def is_button_line(line):
    return line.startswith('Seat ') and ' is the button' in line

def get_button_seat(line):
    return int(line.split(' ')[1])

def is_seat_announcement_line(line):
    line_split = line.split(' ')
    return line_split[0] == 'Seat' and line_split[1].endswith(':') and line_split[-1].startswith('(') and line_split[-1].endswith(').\n')

def get_seat_user_stack(line):
    name = line.split(': ')[1].split(' (')[0]
    line_split = line.split(' ')
    return {'position': int(line_split[1].replace(':', '')), 
            'username': name, 
            'stack': float(line_split[-1].replace('(', '').replace(').', '')),
            'actions': [],
            'cards': [],
            'chips_bet': 0.0}

def is_small_blind_line(line):
    return ' has small blind (' in line

def is_big_blind_line(line):
    return ' has big blind (' in line

def get_blind_amount(line):
    return float(line.split('(')[-1].replace(')', ''))

def get_user_from_player_line(line):
    return line.split(' ')[1]

def is_received_unknown_card_line(line):
    return line.startswith('Player ') and ' received a card.' in line

def is_received_known_card_line(line):
    return line.startswith('Player ') and ' received card: ' in line

def get_player_and_card_from_line(line):
    player = get_user_from_player_line(line)
    card = line.split('[')[-1].replace(']\n', '')
    return player, card

def is_raise_line(line):
    return line.startswith('Player ') and (' raises (' in line or ' allin (' in line or ' bets (' in line or ' caps (' in line)

def get_player_and_raise_amount(line):
    player = get_user_from_player_line(line)
    amount = float(line.split('(')[-1].replace(')', ''))
    return player, amount

def is_fold_line(line):
    return line.startswith('Player ') and ' folds' in line

def is_uncalled_bet_line(line):
    return line.startswith('Uncalled bet (') and ' returned to ' in line

def get_player_and_uncalled_bet_amount(line):
    player = line.split(' ')[-1].replace('\n', '')
    return player, float(line.split(' ')[2].replace(')', '').replace('(', ''))

def is_muck_line(line):
    return line.startswith('Player ') and ' mucks cards' in line

def is_summary_line(line):
    return '------ Summary ------' in line

def is_call_line(line):
    return line.startswith('Player ') and (' calls (' in line or ' straddle (' in line)

def get_player_and_call_amount(line):
    player = get_user_from_player_line(line)
    amount = float(line.split('(')[-1].replace(')', ''))
    return player, amount

def is_wait_line(line):
    return line.startswith('Player ') and (' wait' in line or 'posts' in line or 'sitting out' in line or 'timed out' in line)

def is_river_line(line):
    return line.startswith('*** RIVER ***')

def get_river_card(line):
    return line.split('[')[-1].replace(']\n', '')

def is_turn_line(line):
    return line.startswith('*** TURN ***')

def get_turn_card(line):
    return line.split('[')[-1].replace(']\n', '')

def is_check_line(line):
    return line.startswith('Player ') and ' checks' in line

def is_flop_line(line):
    return line.startswith('*** FLOP ***')

def get_flop_cards(line):
    return line.split('[')[-1].replace(']\n', '').split(' ')



df = pd.DataFrame(columns=['game_id', 'file', 'button_seat', 'small_blind_position', 
                           'small_blind_amount', 'big_blind_amount', 'username', 
                           'position', 'card_1', 'card_2', 'action', 'round_bet', 
                           'round', 'stack_before_action', 'pot_before_action', 
                           'max_bet_before_action', 'player_bet_before_action', 
                           'flop_1', 'flop_2', 'flop_3', 'turn', 'river'])


for file_path in ["data/File196.txt", "data/File198.txt", "data/File199.txt", "data/File200.txt", "data/File201.txt", "data/File203.txt", "data/File204.txt"]:
    with open(file_path, "r") as file:
        lines = file.readlines()
        line_no = 0
        for line in lines:
            if is_start_time_line(line):
                game_id = None
                button_seat = None
                players = []
                small_blind_position = None
                small_blind_amount = None
                big_blind_amount = None
                betting_round = 0
                max_bet = 0.0
                is_post_summary = False
                pot = 0.0
                flop_1 = None
                flop_2 = None
                flop_3 = None
                turn = None
                river = None
                print(line)
                is_post_summary = False
            elif is_game_id_line(line):
                game_id = get_game_id(line)
                print(line)
            elif is_button_line(line):
                button_seat = get_button_seat(line)
                print(line)
            elif is_seat_announcement_line(line):
                players.append(get_seat_user_stack(line))
                print(line)
            elif is_small_blind_line(line):
                small_blind_amount = get_blind_amount(line)
                max_bet = small_blind_amount
                pot += small_blind_amount
                username = get_user_from_player_line(line)
                for player in players:
                    if player['username'] == username:
                        small_blind_position = player['position']
                        player['stack'] -= small_blind_amount
                        player['chips_bet'] += small_blind_amount
                print(line)
            elif is_big_blind_line(line):
                big_blind_amount = get_blind_amount(line)
                max_bet = big_blind_amount
                pot += big_blind_amount
                username = get_user_from_player_line(line)
                for player in players:
                    if player['username'] == username:
                        big_blind_position = player['position']
                        player['stack'] -= big_blind_amount
                        player['chips_bet'] += big_blind_amount
                print(line)
            elif is_received_unknown_card_line(line):
                print(line)
            elif is_received_known_card_line(line):
                username, card = get_player_and_card_from_line(line)
                for player in players:
                    if player['username'] == username:
                        player['cards'].append(card)
                print(line)
            elif is_raise_line(line):
                username, amount = get_player_and_raise_amount(line)
                for player in players:
                    if player['username'] == username:
                        player['actions'].append({'action': 'raise', 
                                                'round_bet': amount,
                                                'round': betting_round,
                                                'stack_before_action': player['stack'],
                                                'pot_before_action': pot,
                                                'max_bet_before_action': max_bet,
                                                'player_bet_before_action': player['chips_bet']})
                            
                        player['stack'] -= amount
                        pot += amount
                        player['chips_bet'] += amount
                        if player['chips_bet'] > max_bet:
                            max_bet = player['chips_bet']
                print(line)
            elif is_fold_line(line):
                username = get_user_from_player_line(line)
                for player in players:
                    if player['username'] == username:
                        player['actions'].append({'action': 'fold', 
                                                'round_bet': 0,
                                                'round': betting_round,
                                                'stack_before_action': player['stack'],
                                                'pot_before_action': pot,
                                                'max_bet_before_action': max_bet,
                                                'player_bet_before_action': player['chips_bet']})
                print(line)
            elif is_call_line(line):
                username, amount = get_player_and_call_amount(line)
                for player in players:
                    if player['username'] == username:
                        player['actions'].append({'action': 'call', 
                                                'round_bet': amount,
                                                'round': betting_round,
                                                'stack_before_action': player['stack'],
                                                'pot_before_action': pot,
                                                'max_bet_before_action': max_bet,
                                                'player_bet_before_action': player['chips_bet']})
                            
                        player['stack'] -= amount
                        pot += amount
                        player['chips_bet'] += amount
                print(line)
            elif is_flop_line(line):
                flop_cards = get_flop_cards(line)
                flop_1 = flop_cards[0]
                flop_2 = flop_cards[1]
                flop_3 = flop_cards[2]
                print(line)
            elif is_check_line(line):
                username = get_user_from_player_line(line)
                for player in players:
                    if player['username'] == username:
                        player['chips_bet'] = max_bet
                        player['actions'].append({'action': 'check', 
                                                'round_bet': 0,
                                                'round': betting_round,
                                                'stack_before_action': player['stack'],
                                                'pot_before_action': pot,
                                                'max_bet_before_action': max_bet,
                                                'player_bet_before_action': player['chips_bet']})
            elif is_turn_line(line):
                turn = get_turn_card(line)
                print(line)
            elif is_river_line(line):
                river = get_river_card(line)
                print(line)
            elif is_uncalled_bet_line(line):
                print(line)
            elif is_muck_line(line):
                print(line)
            elif is_wait_line(line):
                print(line)
            elif is_summary_line(line):
                print(line)
                is_post_summary = True
                for player in players:
                    if len(player['cards']) == 2:
                        for action in player['actions']:
                            record = {
                                'game_id': game_id,
                                'file': file_path,
                                'button_seat': button_seat,
                                'small_blind_position': small_blind_position,
                                'small_blind_amount': small_blind_amount,
                                'big_blind_amount': big_blind_amount,
                                'username': player['username'],
                                'position': player['position'],
                                'card_1': player['cards'][0],
                                'card_2': player['cards'][1],
                                'action': action['action'],
                                'round_bet': action['round_bet'],
                                'round': action['round'],
                                'stack_before_action': action['stack_before_action'],
                                'pot_before_action': action['pot_before_action'],
                                'max_bet_before_action': action['max_bet_before_action'],
                                'player_bet_before_action': action['player_bet_before_action'],
                                'flop_1': flop_1,
                                'flop_2': flop_2,
                                'flop_3': flop_3,
                                'turn': turn,
                                'river': river
                            }
                            df = pd.concat([df, pd.DataFrame([record])])
            elif is_post_summary:
                print(line)
            else:
                raise ValueError(f"Unknow line #{line_no}: {line}")
            line_no += 1
            
df.to_csv('data/poker_games_IlxxxlI_actions.csv', index=False)