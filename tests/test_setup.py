import pytest


def test_setup(GPIO, sn3218, trilobot):
    _ = trilobot.Trilobot()


def test_i2c_warning(GPIO, sn3218, trilobot):
    sn3218.SN3218.side_effect = FileNotFoundError()
    with pytest.raises(RuntimeError):
        _ = trilobot.Trilobot()
    sn3218.SN3218.side_effect = None
