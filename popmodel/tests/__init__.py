from nose.tools import *
import popmodel.main

def test_basic():
    k = popmodel.main.KineticsRun()
    assert_equal(k.sweep.stype, 'sin')
