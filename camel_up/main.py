from loguru import logger

from camel_up.game import Camel, CamelUpGame, Player
from camel_up.strategies import AlwaysRollStrategy, TakeLeaderBetSlipStrategy


def main() -> None:
    players: list[Player] = [Player(AlwaysRollStrategy()), Player(AlwaysRollStrategy()), Player(TakeLeaderBetSlipStrategy()), Player(TakeLeaderBetSlipStrategy())]
    camels: list[Camel] = [
        Camel(color=color) for color in ["red", "blue", "green", "purple", "yellow"]
    ]

    game: CamelUpGame = CamelUpGame(camels, players)

    while not game.is_game_finished():
        game.run_leg()

    for num, player in enumerate(game.players):
        logger.info(f"Player {num} scored {player.coins} points.")


if __name__ == "__main__":
    main()
