import pytest
import popmodel as pm

# test requires HITRAN parameter file, which is not distributed with popmodel.
# assume located at `tests/13_hit12.par` relative path, or can pass in
# alternate path through command line
def pytest_addoption(parser):
    parser.addoption('--hitfile', action='store', help='path to HITRAN file',
            default='tests/13_hit12.par')

@pytest.fixture(scope='session')
def hpar(request):
    return pm.loadhitran.processhitran(request.config.getoption('--hitfile'))

