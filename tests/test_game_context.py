from camel_up.game import Action, Camel, GameContext, Player
from camel_up.actions import RollDiceAction, TakeBettingSlipAction


def test_game_context_camels_move_as_expected_one_camel():
    camels = [Camel(color) for color in ["red"]]
    context = GameContext(camels)

    context.track[1] = ["red"]
    context.current_space["red"] = 1

    context._move_camel("red", 2)
    assert context.track[1] == []
    assert context.track[3] == ["red"]
    assert context.current_space["red"] == 3


def test_game_context_camels_move_as_expected_stacked_camels_lower_moves():
    camels = [Camel(color) for color in ["red", "blue"]]
    context = GameContext(camels)

    context.track[1] = ["red", "blue"]
    context.current_space["red"] = 1
    context.current_space["blue"] = 1

    context._move_camel("blue", 2)
    assert context.track[1] == []
    assert context.track[3] == ["red", "blue"]
    assert context.current_space["red"] == 3
    assert context.current_space["blue"] == 3


def test_game_context_camels_move_as_expected_stacked_camels_upper_moves():
    camels = [Camel(color) for color in ["red", "blue"]]
    context = GameContext(camels)

    context.track[1] = ["red", "blue"]
    context.current_space["red"] = 1
    context.current_space["blue"] = 1

    context._move_camel("red", 2)
    assert context.track[1] == ["blue"]
    assert context.track[3] == ["red"]
    assert context.current_space["red"] == 3
    assert context.current_space["blue"] == 1


def test_betting_slips_set_up_correctly():
    camels = [Camel("red")]
    context = GameContext(camels)

    assert len(context.betting_slips["red"]) == 4
    assert context.betting_slips["red"][0].winnings_if_true == 5
    assert context.betting_slips["red"][1].winnings_if_true == 3
    assert context.betting_slips["red"][2].winnings_if_true == 2
    assert context.betting_slips["red"][3].winnings_if_true == 2


def test_is_leg_finished_returns_true_when_pyramid_is_empty():
    camels = [Camel("red")]
    context = GameContext(camels)
    context.track[1] = ["red"]
    context.current_space["red"] = 1
    assert context.is_leg_finished() is False
    context.roll_dice_and_move_camel()
    assert context.is_leg_finished() is True


def test_is_leg_finished_returns_true_when_camel_is_passed_finishing_line():
    camels = [Camel("red")]
    context = GameContext(camels)

    context.track[6] = ["red"]
    context.current_space["red"] = 6
    assert context.is_leg_finished() is False
    context.track[17] = ["red"]
    context.current_space["red"] = 17
    assert context.is_leg_finished() is True


def test_get_leg_winner():
    camels = [Camel("red"), Camel("blue"), Camel("purple")]
    context = GameContext(camels)

    context.track[6] = ["red"]
    context.current_space["red"] = 6

    context.track[7] = ["blue", "purple"]
    context.current_space["blue"] = 7
    context.current_space["purple"] = 7

    winner = context.get_leg_winner()
    assert winner == "blue"

    runner_up = context.get_leg_runner_up()
    assert runner_up == "purple"


def test_get_leg_runner_up_different_spaces():
    camels = [Camel("red"), Camel("blue"), Camel("purple")]
    context = GameContext(camels)

    context.track[6] = ["red"]
    context.current_space["red"] = 6

    context.track[7] = ["blue"]
    context.current_space["blue"] = 7
    context.track[8] = ["purple"]
    context.current_space["purple"] = 8

    winner = context.get_leg_winner()
    assert winner == "purple"

    runner_up = context.get_leg_runner_up()
    assert runner_up == "blue"


def test_is_game_finished():
    camels = [Camel("red"), Camel("blue"), Camel("purple")]
    context = GameContext(camels)

    context.track[6] = ["red"]
    context.current_space["red"] = 6

    context.track[7] = ["blue"]
    context.current_space["blue"] = 7
    context.track[8] = ["purple"]
    context.current_space["purple"] = 8

    assert not context.is_game_finished()

    context.track[19] = ["purple"]
    context.current_space["purple"] = 19

    assert context.is_game_finished()


def test_player_can_take_move_action(mocker):
    camels = [Camel("red"), Camel("blue"), Camel("purple")]
    context = GameContext(camels)
    context.track[1] = ["red", "blue", "purple"]
    context.current_space["red"] = 1
    context.current_space["blue"] = 1
    context.current_space["purple"] = 1

    player = mocker.MagicMock()
    context.take_action(RollDiceAction(), player)
    player.gain_coins.assert_called_once_with(1)


def test_player_can_take_betting_slip_action(mocker):
    camels = [Camel("red"), Camel("blue"), Camel("purple")]
    context = GameContext(camels)

    assert len(context.betting_slips["red"]) == 4
    player = mocker.MagicMock()

    context.take_action(TakeBettingSlipAction("red"), player)
    player.take_betting_slip.assert_called_once()
    assert len(context.betting_slips["red"]) == 3
