from enum import Enum


class Action(Enum):
    ROLL_DICE = "roll_dice"
    PLACE_TOKEN = "place_token"
    BET_ON_LEG_WINNER = "bet_on_leg_winner"
    BET_ON_OVERALL_WINNER = "bet_on_overall_winner"
    BET_ON_OVERALL_LOSER = "bet_on_overall_loser"
