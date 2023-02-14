from camel_up.game import Camel, CamelUpGame, Player
from camel_up.strategies import AlwaysRollStrategy


def main() -> None:
    players: list[Player] = [Player(AlwaysRollStrategy()) for _ in range(4)]
    camels: list[Camel] = [
        Camel(color=color) for color in ["red", "blue", "green", "purple", "yellow"]
    ]

    game: CamelUpGame = CamelUpGame(camels, players)

    game.run_leg()
    print(game)


if __name__ == "__main__":
    main()
