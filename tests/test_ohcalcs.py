import pytest
import popmodel.ohcalcs as oh

def test_calculateuv_provides_result():
    assert oh.calculateuv(1, 0, 1, 3447) == 32474.62 - 3447

def test_peakpower_provides_result():
    assert oh.peakpower(1, 1, 1) == 1

def test_peakpower_valueerror_none_pulsewidth():
    with pytest.raises(ValueError):
        oh.peakpower(1, None, 1)

def test_peakpower_valueerror_none_reprate():
    with pytest.raises(ValueError):
        oh.peakpower(1, 1, None)
