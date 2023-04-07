from camel_up.actions import Action, RollDiceAction, TakeBettingSlipAction
from camel_up.game import GameContext, Player, PlayerStrategy


class AlwaysRollStrategy(PlayerStrategy):
    """This strategy will always roll the dice, no matter what
    the game context.

    This is intended to be a baseline strategy that other
    strategies can be compared to.
    """

    def choose_action(self, context: GameContext) -> Action:
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
            return TakeBettingSlipAction(current_leader)
        
        return RollDiceAction()