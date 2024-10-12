from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from loguru import logger

from camel_up.actions import Action, ActionType


class CamelUpGame:
    camels: list[Camel]
    players: list[Player]
    finishing_space: int = 17
    player_turn: Player

    _game_context: GameContext

    def __init__(self, camels: list[Camel], players: list[Player]) -> None:
        self.camels = camels
        self.players = players

        self._game_context = GameContext(camels, self.finishing_space)
        self._place_camels_on_board()
        self.player_turn = None

    def run_leg(self) -> None:
        logger.info("Starting leg run")
        while not self._game_context.is_leg_finished():
            player_to_act = self.get_next_player()
            player_action = player_to_act.choose_action(self._game_context)
            self._game_context.take_action(player_action, player_to_act)

        self._score_leg()
        if not self.is_game_finished():
            self._game_context.reset_for_next_leg()

    def run_stepped_turn(self) -> None:
        logger.info("Running a turn...")

        for _ in range(len(self.players)):
            player_to_act = self.get_next_player()
            if not player_to_act.automated:
                return

            player_action = player_to_act.choose_action(self._game_context)
            self._game_context.take_action(player_action, player_to_act)

            if self._game_context.is_leg_finished():
                self._score_leg()
                self._game_context.reset_for_next_leg()
                return

    def is_game_finished(self) -> bool:
        if not self._game_context.is_game_finished():
            return False

        logger.info("Final Scores")
        for i, player in enumerate(self.players):
            logger.info(f"Player {i}: {player.coins}")

        return True

    def get_next_player(self):
        self.player_turn = self.players[
            self._game_context.action_number % len(self.players)
        ]
        logger.info(f"It's now Player {self.player_turn.player_number}'s turn...")
        return self.player_turn

    def _place_camels_on_board(self) -> None:
        for camel in self.camels:
            dice_roll: int = camel.roll_dice()
            self._game_context.current_space[camel.color] = dice_roll
            self._game_context.track[dice_roll] = [
                camel.color
            ] + self._game_context.track[dice_roll]

    def _score_leg(self) -> None:
        logger.info("Scoring leg...")
        winner: str = self._game_context.get_leg_winner()
        runner_up: str = self._game_context.get_leg_runner_up()
        for i, player in enumerate(self.players):
            logger.info(f"Player {i}")
            for slip in player.betting_slips:
                if slip.color == winner:
                    player.gain_coins(slip.winnings_if_true)
                    logger.info(
                        f"Bet on the winner & received {slip.winnings_if_true} coins."
                    )
                elif slip.color == runner_up:
                    player.gain_coins(1)
                    logger.info(f"Bet on the runner-up & received 1 coin.")
                else:
                    player.lose_coins(1)
                    logger.info("Bet on the wrong horse and lost 1 coin.")
            player.return_all_betting_slips()

        logger.info("Current Scores")
        for i, player in enumerate(self.players):
            logger.info(f"Player {i}: {player.coins}")

    def reset(self) -> None:
        self._game_context = GameContext(self.camels, self.finishing_space)
        [player.reset() for player in self.players]
        self._place_camels_on_board()

    def get_winner(self) -> Player:
        highest_score = 0
        highest_score_player = None
        for player in self.players:
            if player.coins > highest_score:
                highest_score_player = player
                highest_score = player.coins

        return highest_score_player

    def get_leg_number(self) -> int:
        return self._game_context.leg_number


class GameContext:
    camels: list[Camel]
    betting_slips: dict[str, list[BettingSlip]]
    track: dict[int, list[str]]
    current_space: dict[str, int]
    finishing_space: int
    action_number: int
    _pyramid: Pyramid

    def __init__(self, camels: list[Camel], finishing_space: int = 17) -> None:
        self.camels = camels
        self.camel_colors: list[str] = [camel.color for camel in self.camels]
        self.betting_slips = self._set_up_betting_slips()
        self._pyramid = Pyramid([camel.dice for camel in camels])
        self.track = defaultdict(list)
        self.current_space = {}
        self.finishing_space = finishing_space
        self.action_number = 0
        self.leg_number = 1

    def _set_up_betting_slips(self) -> dict[str, list[BettingSlip]]:
        output: dict[str, list[BettingSlip]] = {}
        for camel in self.camels:
            output[camel.color] = []
            output[camel.color].append(BettingSlip(camel.color, 5))
            output[camel.color].append(BettingSlip(camel.color, 3))
            output[camel.color].append(BettingSlip(camel.color, 2))
            output[camel.color].append(BettingSlip(camel.color, 2))
        return output

    def take_action(self, player_action: Action, player: Player, **kwargs) -> None:
        match player_action.action_type:
            case ActionType.ROLL_DICE:
                self.roll_dice_and_move_camel()
                player.gain_coins(1)
            case ActionType.BET_ON_LEG_WINNER:
                player.take_betting_slip(self.get_top_betting_slip(player_action.color))

        self.action_number += 1

    def is_leg_finished(self) -> bool:
        logger.info(
            f"Leg finish check! Dice still to roll are {self._pyramid.dice_still_to_roll}"
        )
        return len(self._pyramid.dice_still_to_roll) == 0 or self.is_game_finished()

    def is_game_finished(self) -> bool:
        return max(self.current_space.values()) >= self.finishing_space

    def roll_dice_and_move_camel(self):
        color, dice_roll = self._pyramid.roll_dice()
        logger.info(f"Moving {color} {dice_roll} spaces.")
        self._move_camel(color, dice_roll)

    def get_leg_winner(self) -> str:
        furthest_space: int = max(self.current_space.values())
        return self.track[furthest_space][0]

    def get_leg_runner_up(self) -> str:
        if len(self.camels) < 2:
            return ""

        spaces = list(self.current_space.values())
        spaces.sort(reverse=True)
        second_placed_camel_space = spaces[1]

        if second_placed_camel_space == max(spaces):
            return self.track[second_placed_camel_space][1]

        return self.track[second_placed_camel_space][0]

    def get_top_betting_slip(self, color: str) -> BettingSlip:
        return self.betting_slips[color].pop(0)

    def reset_for_next_leg(self) -> None:
        self._pyramid.reset()
        self.betting_slips = self._set_up_betting_slips()
        self.leg_number += 1

    def _move_camel(self, color: str, dice_roll: int):
        current_position = self.current_space[color]
        current_index = self.track[current_position].index(color)
        stack = self.track[current_position][0 : current_index + 1]
        self.track[current_position] = self.track[current_position][current_index + 1 :]
        self.track[current_position + dice_roll] = (
            stack + self.track[current_position + dice_roll]
        )
        for camel_color in stack:
            self.current_space[camel_color] = current_position + dice_roll

    def get_current_occupied_spaces(self) -> list[int]:
        output = list(set(self.current_space.values()))
        output = sorted(output)
        return output


class Pyramid:
    dice: list[Dice]
    dice_already_rolled: list[Dice]
    dice_still_to_roll: list[Dice]

    def __init__(self, dice: list[Dice]) -> None:
        self.dice = dice
        self.reset()

    def roll_dice(self) -> tuple[str, int]:
        dice_to_roll = self.dice_still_to_roll.pop()
        self.dice_already_rolled.append(dice_to_roll)
        return (dice_to_roll.color, dice_to_roll.roll())

    def reset(self) -> None:
        self.dice_still_to_roll = copy.copy(self.dice)
        self.dice_already_rolled = []
        random.shuffle(self.dice_still_to_roll)


class Dice:
    color: str
    possible_values: list[int] = [1, 2, 3]

    def __init__(self, color: str) -> None:
        self.color = color

    def __str__(self) -> str:
        return f"Dice: {self.color}"

    def __repr__(self):
        return self.__str__()

    def roll(self) -> int:
        return random.sample(self.possible_values, k=1)[0]


class Camel:
    dice: Dice
    color: str

    def __init__(self, color: str) -> None:
        self.color = color
        self.dice = Dice(color)

    def roll_dice(self) -> int:
        return self.dice.roll()


class Player:
    strategy: PlayerStrategy
    coins: int
    betting_slips: list[BettingSlip]
    automated: bool
    player_number: int

    def __init__(
        self, strategy: PlayerStrategy, player_number=1, automated: bool = True
    ) -> None:
        self.coins = 3
        self.strategy = strategy
        self.betting_slips = []
        self.automated = automated
        self.player_number = player_number

    def gain_coins(self, coins_to_gain: int) -> None:
        self.coins += coins_to_gain

    def lose_coins(self, coins_to_lose: int) -> None:
        self.coins = max(0, self.coins - coins_to_lose)

    def choose_action(self, context: GameContext) -> Action:
        return self.strategy.choose_action(context)

    def take_betting_slip(self, betting_slip: BettingSlip) -> None:
        self.betting_slips.append(betting_slip)

    def return_all_betting_slips(self) -> list[BettingSlip]:
        betting_slips = [a for a in self.betting_slips]
        self.betting_slips = []
        return betting_slips

    def reset(self) -> None:
        self.coins = 3
        self.betting_slips = []


@dataclass
class BettingSlip:
    color: str
    winnings_if_true: int


class PlayerStrategy(ABC):
    """Base class for player strategy. This will
    be the place to define the logic of what action to take
    given the current game context.

    """

    @abstractmethod
    def choose_action(self, context: GameContext) -> Action: ...
