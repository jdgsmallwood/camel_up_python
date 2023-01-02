from __future__ import annotations
from dataclasses import dataclass

import random


class CamelUpGame:

    camels: list[Camel]
    players: list[Player]
    _pyramid: Pyramid
    _game_context: GameContext

    def __init__(self, camels: list[Camel], players: list[Player]) -> None:
        self.camels = camels
        self.players = players
        self._pyramid = Pyramid([camel.dice for camel in camels])
        self._game_context = GameContext(camels)


class GameContext:
    
    camels: list[Camel]
    betting_slips: dict[str, list[BettingSlip]]
    track: dict[int, Camel]
    current_space: dict[str, int]

    def __init__(self, camels: list[Camel]) -> None:
        self.camels = camels
        self.betting_slips = self._set_up_betting_slips()
    
    def _set_up_betting_slips(self) -> dict[str, list[BettingSlip]]:
        output: dict[str, list[BettingSlip]] = {}
        for camel in self.camels:
            output[camel.color] = []
            output[camel.color].append(BettingSlip(camel.color, 5))
            output[camel.color].append(BettingSlip(camel.color, 3))
            output[camel.color].append(BettingSlip(camel.color, 2))
            output[camel.color].append(BettingSlip(camel.color, 2))
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

    def __init__(self, strategy: PlayerStrategy) -> None:
        self.coins = 3
        self.strategy = strategy
    
    def gain_coins(self, coins_to_gain: int) -> None:
        self.coins += coins_to_gain

    def lose_coins(self, coins_to_lose: int) -> None:
        self.coins = max(0, self.coins - coins_to_lose)

    def choose_action(self, context: GameContext) -> Action:
        return self.strategy.choose_action(context)


@dataclass
class BettingSlip:
    color: str
    winnings_if_true: int


class Action:
    pass

class PlayerStrategy:
    
    def choose_action(self, context: GameContext) -> Action:
        return Action()
