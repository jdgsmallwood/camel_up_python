from camel_up.actions import RollDiceAction, TakeBettingSlipAction
from camel_up.strategies import AlwaysRollStrategy, TakeLeaderBetSlipStrategy
from camel_up.game import BettingSlip, GameContext, Camel


def test_always_rolls_dice():
    context = None

    assert isinstance(AlwaysRollStrategy().choose_action(context), RollDiceAction)


def test_take_leader_bet_slip_strategy():
    camels = [Camel("red"), Camel("blue"), Camel("purple")]
    context = GameContext(camels)

    context.current_space = {
        "red": 2,
        "blue": 4,
        "purple": 8,
    }

    context.track = {2: ["red"], 4: ["blue"], 8: ["purple"]}
    chosen_action = TakeLeaderBetSlipStrategy().choose_action(context)
    assert isinstance(chosen_action, TakeBettingSlipAction)
    assert chosen_action.color == "purple"


def test_take_leader_bet_slip_strategy_rolls_dice_when_no_more_available():
    camels = [Camel("red"), Camel("blue"), Camel("purple")]
    context = GameContext(camels)

    context.current_space = {
        "red": 2,
        "blue": 4,
        "purple": 8,
    }

    context.track = {2: ["red"], 4: ["blue"], 8: ["purple"]}

    context.betting_slips = {
        "red": [BettingSlip("red", 5)],
        "blue": [BettingSlip("blue", 2)],
        "purple": [],
    }
    chosen_action = TakeLeaderBetSlipStrategy().choose_action(context)
    assert isinstance(chosen_action, RollDiceAction)
