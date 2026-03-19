from services.identity_policy import is_same_position


def test_is_same_position_true_at_small_shift():
    assert bool(is_same_position([0, 0, 100, 200], [10, 5, 110, 205]))


def test_is_same_position_false_when_far_away():
    assert not bool(is_same_position([0, 0, 100, 200], [150, 150, 250, 350]))


def test_is_same_position_false_on_boundary_distance():
    # Radius is 60; center shift is exactly 60 => strict '<' means False
    assert not bool(is_same_position([0, 0, 100, 200], [60, 0, 160, 200]))
