import pytest
import popmodel as pm
from pkg_resources import resource_filename

@pytest.fixture(scope='session')
def par():
    yamlpath = resource_filename('popmodel','data/parameters_template.yaml')
    return pm.importyaml(yamlpath)

@pytest.fixture(scope='function')
def k(hpar,par):
    return pm.KineticsRun(hpar,**par)

def test_KineticsRun_instance_has_attributes(k):
    assert hasattr(k, 'detcell')
    assert hasattr(k, 'irlaser')
    assert hasattr(k, 'uvlaser')
    assert hasattr(k, 'odepar')
    assert hasattr(k, 'irline')
    assert hasattr(k, 'uvline')
    assert hasattr(k, 'rates')
