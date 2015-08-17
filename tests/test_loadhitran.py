import pytest
import popmodel as pm
from pkg_resources import resource_filename

@pytest.fixture(scope='module')
def hpath():
    hpath = resource_filename('popmodel', 'data/hitran_sample.par')
    return hpath

def test_processhitran_runs_without_error(hpath):
    pm.loadhitran.processhitran(hpath)
