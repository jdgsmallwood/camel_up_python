from camel_up.actions import Action
from camel_up.strategies import AlwaysRollStrategy


def test_always_rolls_dice():
    context = None

    assert AlwaysRollStrategy().choose_action(context) == Action.ROLL_DICE
