from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from camel_up.actions import Action


class CamelUpGame:

    camels: list[Camel]
    players: list[Player]
    finishing_space: int = 17

    _game_context: GameContext

    def __init__(self, camels: list[Camel], players: list[Player]) -> None:
        self.camels = camels
        self.players = players

        self._game_context = GameContext(camels, self.finishing_space)
        self._place_camels_on_board()

    def run_leg(self) -> None:
        while not self._game_context.is_leg_finished():
            for player in self.players:
                player_action = player.choose_action(self._game_context)
                self._game_context.take_action(player_action, player)
                if self._game_context.is_leg_finished():
                    break

    def _place_camels_on_board(self) -> None:
        for camel in self.camels:
            dice_roll: int = camel.roll_dice()
            self._game_context.current_space[camel.color] = dice_roll
            self._game_context.track[dice_roll] = [
                camel.color
            ] + self._game_context.track[dice_roll]


class GameContext:

    camels: list[Camel]
    betting_slips: dict[str, list[BettingSlip]]
    track: dict[int, list[str]]
    current_space: dict[str, int]
    finishing_space: int
    _pyramid: Pyramid

    def __init__(self, camels: list[Camel], finishing_space: int = 17) -> None:
        self.camels = camels
        self.camel_colors: list[str] = [camel.color for camel in self.camels]
        self.betting_slips = self._set_up_betting_slips()
        self._pyramid = Pyramid([camel.dice for camel in camels])
        self.track = defaultdict(list)
        self.current_space = {}
        self.finishing_space = finishing_space

    def _set_up_betting_slips(self) -> dict[str, list[BettingSlip]]:
        output: dict[str, list[BettingSlip]] = {}
        for camel in self.camels:
            output[camel.color] = []
            output[camel.color].append(BettingSlip(camel.color, 5))
            output[camel.color].append(BettingSlip(camel.color, 3))
            output[camel.color].append(BettingSlip(camel.color, 2))
            output[camel.color].append(BettingSlip(camel.color, 2))
        return output

    def take_action(self, player_action: Action, player: Player) -> None:
        match player_action:
            case Action.ROLL_DICE:
                self.roll_dice_and_move_camel()
                player.gain_coins(1)

    def is_leg_finished(self) -> bool:
        return len(self._pyramid.dice_still_to_roll) == 0 or any(
            self.current_space[color] >= self.finishing_space
            for color in self.camel_colors
        )

    def roll_dice_and_move_camel(self):
        color, dice_roll = self._pyramid.roll_dice()
        self._move_camel(color, dice_roll)

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
        self.dice_still_to_roll = self.dice
        self.dice_already_rolled = []
        random.shuffle(self.dice_still_to_roll)


class Dice:
    color: str
    possible_values: list[int] = [1, 2, 3]

    def __init__(self, color: str) -> None:
        self.color = color

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

    def __init__(self, strategy: PlayerStrategy) -> None:
        self.coins = 3
        self.strategy = strategy
        self.betting_slips = []

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
    def choose_action(self, context: GameContext) -> Action:
        ...
