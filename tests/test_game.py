from camel_up.game import (
    Action,
    BettingSlip,
    Camel,
    Dice,
    GameContext,
    Player,
    PlayerStrategy,
    Pyramid,
)


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
        player = Player(PlayerStrategy())
        assert player.coins == 3

    def test_gain_coins(self):
        player = Player(PlayerStrategy())
        player.gain_coins(2)
        assert player.coins == 5

    def test_lose_coins(self):
        player = Player(PlayerStrategy())
        player.lose_coins(2)
        assert player.coins == 1

    def test_lose_coins_with_floor_at_zero(self):
        player = Player(PlayerStrategy())
        player.lose_coins(4)
        assert player.coins == 0

    def test_gain_betting_slip_and_return_all(self):
        player = Player(PlayerStrategy())
        player.take_betting_slip(BettingSlip("red", 2))
        assert len(player.betting_slips) == 1
        player.return_all_betting_slips()
        assert len(player.betting_slips) == 0

    def test_choose_action(self):
        player = Player(PlayerStrategy())
        camels = [Camel(color) for color in ["red"]]
        context = GameContext(camels)
        assert isinstance(player.choose_action(context), Action)
