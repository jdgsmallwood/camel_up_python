from camel_up.actions import RollDiceAction
from camel_up.strategies import AlwaysRollStrategy


def test_always_rolls_dice():
    context = None

    assert isinstance(AlwaysRollStrategy().choose_action(context), RollDiceAction)
