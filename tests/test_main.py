import pytest
import popmodel as pm
from pkg_resources import resource_filename

@pytest.fixture(scope='session')
def hpar():
    hpath = resource_filename('popmodel', 'data/hitran_sample.par')
    return pm.loadhitran.processhitran(hpath)

@pytest.fixture(scope='session')
def par():
    yamlpath = resource_filename('popmodel','data/parameters_template.yaml')
    return pm.importyaml(yamlpath)

# initiated KineticsRun without running solveode
@pytest.fixture(scope='session')
def k(hpar,par):
    return pm.KineticsRun(hpar,**par)

# KineticsRun instance that has had solveode() run.
@pytest.fixture(scope='session')
def k_solved(hpar, par):
    k_solved = pm.KineticsRun(hpar, **par)
    k_solved.solveode()
    return k_solved

# KineticsRun instance with dosweep = True
@pytest.fixture(scope='session')
def k_sweep(hpar, par):
    par_sweep = par.copy()
    par_sweep['sweep']['dosweep'] = True
    k_sweep = pm.KineticsRun(hpar, **par)
    k_sweep.solveode()
    return k_sweep

def test_KineticsRun_instance_has_attributes(k):
    assert hasattr(k, 'detcell')
    assert hasattr(k, 'irlaser')
    assert hasattr(k, 'uvlaser')
    assert hasattr(k, 'odepar')
    assert hasattr(k, 'irline')
    assert hasattr(k, 'uvline')
    assert hasattr(k, 'rates')

def test_raise_error_popsfigure_without_N(k):
    with pytest.raises(AttributeError):
        k.popsfigure()

@pytest.mark.mpl_image_compare
def test_popsfigure_plots_suite_of_subpop_popcodes(k_solved):
    return k_solved.popsfigure(subpop=['ahd','bsl','clp','dda','b'])

@pytest.mark.mpl_image_compare
def test_popsfigure_default_subpop(k_solved):
    return k_solved.popsfigure()

@pytest.mark.mpl_image_compare
def test_vslaserfigure_defaults(k_sweep):
    return k_sweep.vslaserfigure(k_sweep.popseries('b'))

@pytest.mark.mpl_image_compare
def test_absfigure_defaults(k_sweep):
    return k_sweep.absfigure()

def test_calcfluor_returns_value(k_solved):
    assert k_solved.calcfluor()
