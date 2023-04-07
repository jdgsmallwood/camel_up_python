from enum import Enum


class ActionType(Enum):
    ROLL_DICE = "roll_dice"
    PLACE_TOKEN = "place_token"
    BET_ON_LEG_WINNER = "bet_on_leg_winner"
    BET_ON_OVERALL_WINNER = "bet_on_overall_winner"
    BET_ON_OVERALL_LOSER = "bet_on_overall_loser"

class Action:
    action_type: ActionType


class RollDiceAction(Action):
    action_type = ActionType.ROLL_DICE

class TakeBettingSlipAction(Action):
    action_type = ActionType.BET_ON_LEG_WINNER

    def __init__(self, color: str):
        self.color = color
        super().__init__()
