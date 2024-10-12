from camel_up.actions import RollDiceAction
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
        return RollDiceAction()


class TestPyramid:
    def test_can_roll_dice(self):
        dice = [Dice("red")]
        pyramid = Pyramid(dice)

        output = pyramid.roll_dice()
        assert isinstance(output, tuple)
        assert output[0] == "red"
        assert isinstance(output[1], int)

    def test_doesnt_remove_dice_from_pyramid(self):
        dice = [Dice("red")]
        pyramid = Pyramid(dice)
        assert len(pyramid.dice) == 1
        assert len(pyramid.dice_still_to_roll) == 1

        output = pyramid.roll_dice()
        assert len(pyramid.dice) == 1
        assert len(pyramid.dice_already_rolled) == 1


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

    def test_score_leg(self):
        camels = [Camel("red")]
        game = CamelUpGame(camels, [Player(AlwaysRollStrategy())])

        context = GameContext(camels)
        game.players[0].betting_slips.append(BettingSlip("red", 5))
        context.track[1] = ["red"]
        context.current_space["red"] = 1

        game._game_context = context
        game._score_leg()
        assert game.players[0].coins == 8
        # check all betting slips are returned.
        assert game.players[0].betting_slips == []

    def test_is_game_finished_passes_to_game_context_object(self, mocker):
        context_mock = mocker.MagicMock()

        camels = [Camel("red")]
        game = CamelUpGame(camels, [Player(AlwaysRollStrategy())])
        game._game_context = context_mock

        game.is_game_finished()

        context_mock.is_game_finished.assert_called_once()

    def test_after_leg_is_run_game_is_set_up_for_next_leg(self):
        camels = [Camel("red")]
        game = CamelUpGame(camels, [Player(AlwaysRollStrategy())])

        game._game_context.betting_slips["red"] = []
        game.run_leg()
        assert game._game_context._pyramid.dice_already_rolled == []
        assert len(game._game_context.betting_slips["red"]) == 4
        # check all betting slips are returned.
        assert game.players[0].betting_slips == []

    def test_get_next_player_returns_as_expected(self, mocker):
        players = [mocker.MagicMock() for i in range(4)]
        camels = [Camel("red")]
        game = CamelUpGame(camels, players)

        assert game.get_next_player() == players[0]
        game._game_context.action_number = 3
        assert game.get_next_player() == players[3]
        game._game_context.action_number = 6
        assert game.get_next_player() == players[2]

    def test_reset(self, mocker):
        players = [mocker.MagicMock() for i in range(4)]
        camels = [Camel("red")]
        game = CamelUpGame(camels, players)

        game._game_context.current_space["red"] = 400

        game.reset()
        assert game._game_context.current_space["red"] <= 3

    def test_stepped_turn_stops_on_automated_player(self, mocker):
        non_automated_player = mocker.MagicMock()
        non_automated_player.automated = False
        players = [mocker.MagicMock() for i in range(4)] + [non_automated_player]
        camels = [Camel("red")]
        game = CamelUpGame(camels, players)

        game.run_stepped_turn()
        assert not game.player_turn.automated

    def test_leg_number_reported_faithfully(self, mocker):
        camels = [Camel("red")]
        game = CamelUpGame(camels, [Player(AlwaysRollStrategy())])

        game._game_context.track[16] = ["red"]
        # Chosen so that any dice roll will take it over the finish line.
        game._game_context.current_space["red"] = 16
        game._game_context.leg_number = 17
        game.run_leg()
        
        assert game.is_game_finished()
        assert game.get_leg_number() == 17
