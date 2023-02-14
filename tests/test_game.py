from camel_up.game import (
    Action,
    BettingSlip,
    Camel,
    CamelUpGame,
    Dice,
    GameContext,
    Player,
    PlayerStrategy,
    Pyramid,
)
from camel_up.strategies import AlwaysRollStrategy


class MockPlayerStrategy(PlayerStrategy):
    def choose_action(self, context: GameContext) -> Action:
        return Action.ROLL_DICE


class TestPyramid:
    def test_can_roll_dice(self):
        dice = [Dice("red")]
        pyramid = Pyramid(dice)

        output = pyramid.roll_dice()
        assert isinstance(output, tuple)
        assert output[0] == "red"
        assert isinstance(output[1], int)


class TestDice:
    def test_can_roll(self):
        dice = Dice("red")

        assert dice.color == "red"
        assert 1 <= dice.roll() <= 3


class TestCamel:
    def test_instantiation(self):
        camel = Camel("red")
        assert camel.color == "red"

    def test_can_roll(self):
        camel = Camel("red")
        assert 1 <= camel.roll_dice() <= 3


class TestPlayer:
    def test_instantiation(self):
        player = Player(MockPlayerStrategy())
        assert player.coins == 3

    def test_gain_coins(self):
        player = Player(MockPlayerStrategy())
        player.gain_coins(2)
        assert player.coins == 5

    def test_lose_coins(self):
        player = Player(MockPlayerStrategy())
        player.lose_coins(2)
        assert player.coins == 1

    def test_lose_coins_with_floor_at_zero(self):
        player = Player(MockPlayerStrategy())
        player.lose_coins(4)
        assert player.coins == 0

    def test_gain_betting_slip_and_return_all(self):
        player = Player(MockPlayerStrategy())
        player.take_betting_slip(BettingSlip("red", 2))
        assert len(player.betting_slips) == 1
        player.return_all_betting_slips()
        assert len(player.betting_slips) == 0

    def test_choose_action(self):
        player = Player(MockPlayerStrategy())
        camels = [Camel(color) for color in ["red"]]
        context = GameContext(camels)
        assert isinstance(player.choose_action(context), Action)


class TestGame:
    def test_camels_are_put_on_track_initially(self):
        camels = [Camel("red")]
        game = CamelUpGame(camels, [])

        assert "red" in game._game_context.current_space
        assert any(len(game._game_context.track[space]) == 1 for space in range(1, 4))

    def test_can_successfully_run_a_leg(self):
        # Just assert no errors on running this.
        camels = [Camel("red")]
        game = CamelUpGame(camels, [Player(AlwaysRollStrategy())])

        game.run_leg()
