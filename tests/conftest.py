from __future__ import division
import pytest
import popmodel as pm
from pkg_resources import resource_filename

@pytest.fixture(scope='session')
def hpar():
    hpath = resource_filename('popmodel', 'data/hitran_sample.par')
    return pm.loadhitran.processhitran(hpath)
