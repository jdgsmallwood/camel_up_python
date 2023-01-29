from camel_up.game import GameContext, Camel


def test_game_context_camels_move_as_expected_one_camel():
    camels = [Camel(color) for color in ["red"]]
    context = GameContext(camels)

    context.track[1] = ['red']
    context.current_space['red'] = 1

    context._move_camel('red', 2)
    assert context.track[1] == []
    assert context.track[3] == ["red"]
    assert context.current_space['red'] == 3

