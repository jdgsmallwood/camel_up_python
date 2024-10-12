import gymnasium as gym
from gymnasium import spaces
import numpy as np

from camel_up import actions
from camel_up.game import Camel, CamelUpGame, Player
from camel_up.strategies import AlwaysRollStrategy, TakeLeaderBetSlipStrategy
from loguru import logger


class CamelUpEnv(gym.Env):
    def __init__(self):
        self.camels = ["red", "blue", "green", "purple", "yellow"]
        self.n_camels = len(self.camels)
        self.n_players = 4
        self.action_space = spaces.Discrete(
            self.n_camels + 1
        )  # bet slip on each camel + roll dice
        self.action_mask = np.ones(
            shape=(self.n_camels + 1),
        )

        self.observation_space = spaces.Dict(
            {
                "camel_locations": spaces.Dict(
                    {
                        i: spaces.Tuple(
                            [spaces.Box(0, 17), spaces.Box(0, self.n_camels)]
                        )
                        for i in self.camels
                    }
                ),
                "leg_betting_slips_available": spaces.Dict(
                    {i: spaces.Discrete(4) for i in self.camels}
                ),
            }
        )

        self.action_mapping = {
            **{
                self.n_camels: actions.RollDiceAction(),
            },
            **{
                i: actions.TakeBettingSlipAction(color)
                for i, color in enumerate(self.camels)
            },
        }

        self.game = CamelUpGame(
            [Camel(color=color) for color in self.camels],
            [
                Player(AlwaysRollStrategy(), player_number=0),
                Player(AlwaysRollStrategy(), player_number=1),
                Player(TakeLeaderBetSlipStrategy(), player_number=2),
                Player(AlwaysRollStrategy(), automated=False, player_number=3),
            ],
        )

        self.winners = []

    def _get_obs(self):
        occupied_spaces = self.game._game_context.get_current_occupied_spaces()
        camel_locations = {}
        for space in occupied_spaces:
            for i, color in enumerate(self.game._game_context.track[space]):
                camel_locations[color] = (space, i)

        return {
            "camel_locations": camel_locations,
            "leg_betting_slips_available": {
                color: [
                    0
                    for i in range(
                        4 - len(self.game._game_context.betting_slips[color])
                    )
                ]
                + [1 for i in range(len(self.game._game_context.betting_slips[color]))]
                for color in self.camels
            },
        }

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        logger.info("Game is over... resetting for new game.")
        self.game.reset()
        self.action_mask = self._update_action_mask()
        while not self.game.player_turn or self.game.player_turn.automated:
            if self.game.is_game_finished():
                terminate = True
                break
            self.game.run_stepped_turn()
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        if not self.action_mask[action]:
            raise ValueError(f"Action {action} is not allowed in this state!")
        action_to_take = self.action_mapping[action]
        # current_coins = self.game.player_turn.coins
        self.game._game_context.take_action(action_to_take, self.game.player_turn)

        if self.game._game_context.is_leg_finished():
            self.game._score_leg()
            self.game._game_context.reset_for_next_leg()

        terminate = False

        first_run = True
        while first_run or self.game.player_turn.automated:
            if self.game.is_game_finished():
                terminate = True
                winner = self.game.get_winner()
                self.winners.append(winner.player_number)
                break
            self.game.run_stepped_turn()
            first_run = False

        obs = self._get_obs()
        reward = self.game.player_turn.coins  # - current_coins
        self.action_mask = self._update_action_mask()
        return obs, reward, terminate, False, {}

    def _update_action_mask(self):
        return np.array(
            [
                int(len(self.game._game_context.betting_slips[camel.color]) > 0)
                for camel in self.game.camels
            ]
            + [1]
        )
