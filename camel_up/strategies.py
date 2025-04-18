import random

from loguru import logger

from camel_up.actions import Action, RollDiceAction, TakeBettingSlipAction
from camel_up.game import GameContext, Player, PlayerStrategy


class AlwaysRollStrategy(PlayerStrategy):
    """This strategy will always roll the dice, no matter what
    the game context.

    This is intended to be a baseline strategy that other
    strategies can be compared to.
    """

    def choose_action(self, context: GameContext) -> Action:
        logger.info("Player is rolling dice...")
        return RollDiceAction()


class TakeLeaderBetSlipStrategy(PlayerStrategy):
    """This strategy will take the current leader's top bet tile if possible.
    Otherwise will roll the dice.

    This is intended to be a baseline strategy that other
    strategies can be compared to.
    """

    def choose_action(self, context: GameContext) -> Action:
        current_leader = context.get_leg_winner()
        if len(context.betting_slips[current_leader]) > 0:
            logger.info(
                f"Taking betting slip of current leader which is {current_leader}"
            )
            return TakeBettingSlipAction(current_leader)
        logger.info("rolling dice...")
        return RollDiceAction()


class RandomActionStrategy(PlayerStrategy):
    """This strategy chooses randomly between all different actions."""

    def choose_action(self, context: GameContext) -> Action:
        possible_actions = [RollDiceAction()]

        for color in context.camel_colors:
            if len(context.betting_slips[color]) > 0:
                possible_actions.append(TakeBettingSlipAction(color))

        action = random.choice(possible_actions)

        logger.info(f"Choosing {action}...")
        return action
