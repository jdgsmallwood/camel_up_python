from abc import ABC, abstractmethod

from camel_up.actions import Action
from camel_up.game import GameContext, PlayerStrategy


class AlwaysRollStrategy(PlayerStrategy):
    """This strategy will always roll the dice, no matter what
    the game context.

    This is intended to be a baseline strategy that other
    strategies can be compared to.
    """

    def choose_action(self, context: GameContext) -> Action:
        return Action.ROLL_DICE
