from frb.antenna import select_antenna
import pytest


def test_select_antenna():
    antenna = ['RADIO-AS', 'ZELENCHK', 'NOTO', 'USUDA64', 'SVETLOE', 'ARECIBO']
    selected = select_antenna(antenna)
    assert selected == (['USUDA64'], ['NOTO', 'RADIO-AS'])
    selected = select_antenna(antenna, ignored=['RADIO-AS'])
    assert selected == (['USUDA64'], ['NOTO', 'ZELENCHK'])
    selected = select_antenna(antenna, n_big=2, n_small=3,
                                  ignored=['RADIO-AS'])
    assert selected == (['USUDA64', 'ARECIBO'], ['NOTO', 'ZELENCHK', 'SVETLOE'])

    with pytest.raises(Exception):
        select_antenna(antenna, n_big=3)
    with pytest.raises(Exception):
        select_antenna(antenna, n_small=5)

    antenna.append('SOME')
    with pytest.raises(KeyError):
        select_antenna(antenna)
