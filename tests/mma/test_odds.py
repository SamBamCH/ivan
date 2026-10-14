import pytest

from ivan.sports.mma.odds import american_to_prob, to_float, win_split


@pytest.mark.parametrize("odds,prob", [(100, 0.5), (-100, 0.5), (200, 1 / 3), (-150, 0.6), (0, 0.5), (None, 0.5), (-1050, 1050 / 1150)])
def test_american_to_prob(odds, prob):
    assert american_to_prob(odds) == pytest.approx(prob)


def test_to_float():
    assert to_float("+150") == 150
    assert to_float("1,800") == 1800
    assert to_float("") is None
    assert to_float(None) is None
    assert to_float("nan") is None


def test_win_split_normalises():
    a, b = win_split(-340, 270)
    assert a + b == pytest.approx(1.0)
    assert a > b
    assert win_split(None, None) == (0.5, 0.5)
