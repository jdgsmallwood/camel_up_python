from camel_up.game import Camel, GameContext


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
