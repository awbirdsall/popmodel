import pytest
import popmodel.ohcalcs as oh

def test_calculateuv_provides_result():
    assert oh.calculateuv(1, 0, 1, 3447) == 32474.62 - 3447
