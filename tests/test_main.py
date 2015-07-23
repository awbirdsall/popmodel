import pytest
import popmodel as pm
from pkg_resources import resource_filename
from collections import Mapping
from copy import deepcopy

@pytest.fixture(scope='session')
def hpar():
    hpath = resource_filename('popmodel', 'data/hitran_sample.par')
    return pm.loadhitran.processhitran(hpath)

# Use parameters_template YAML file included with package as base set of
# parameters. Toggle settings in deepcopies of par (deepcopy needed for nested
# dict) in tests as needed.
@pytest.fixture(scope='session')
def par():
    yamlpath = resource_filename('popmodel','data/parameters_template.yaml')
    return pm.importyaml(yamlpath)

# KineticsRun instance without running solveode
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
    par_sweep = deepcopy(par)
    par_sweep['sweep']['dosweep'] = True
    k_sweep = pm.KineticsRun(hpar, **par_sweep)
    k_sweep.solveode()
    return k_sweep

# KineticsRun instance set up for UV (vibrationless transition) without IR
@pytest.fixture(scope='session')
def k_uvonly(hpar, par):
    par_uvonly = deepcopy(par)
    uvonly_reqs = {'uvline': {'vib': '00'}, 'odepar': {'withoutUV': False,
                                                       'withoutIR': True}}
    _nestedupdate(par_uvonly, uvonly_reqs)
    k_uvonly = pm.KineticsRun(hpar, **par_uvonly)
    k_uvonly.solveode()
    return k_uvonly

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
def test_popsfigure_integrate_uvlaser_only(k_uvonly):
    return k_uvonly.popsfigure(subpop=['b', 'c'])

@pytest.mark.mpl_image_compare
def test_vslaserfigure_defaults(k_sweep):
    return k_sweep.vslaserfigure(k_sweep.popseries('b'))

@pytest.mark.mpl_image_compare
def test_absfigure_defaults(k_sweep):
    return k_sweep.absfigure()

def test_calcfluor_returns_value(k_solved):
    assert k_solved.calcfluor()

def test_calcfluor_uvonly_returns_positive_value(k_uvonly):
    assert (k_uvonly.calcfluor() > 0)

@pytest.mark.parametrize("par_laser_setup", [
    {'uvline': {'vib': '00'}, 'odepar': {'withoutUV': False,
                                         'withoutIR': True}},
    {'uvline': {'vib': '10'}, 'odepar': {'withoutUV': False,
                                         'withoutIR': True}},
    {'uvline': {'vib': '11'}, 'odepar': {'withoutUV': False,
                                         'withoutIR': False}},
    {'uvline': {'vib': '01'}, 'odepar': {'withoutUV': False,
                                         'withoutIR': False}},
    {'odepar': {'withoutUV': True, 'withoutIR': True}},
    {'odepar': {'withoutUV': True, 'withoutIR': False}},
    ])
def test_KineticsRun_init_toggle_laser_setup(hpar, par, par_laser_setup):
    par_lasers = deepcopy(par)
    _nestedupdate(par_lasers, par_laser_setup)
    k_lasers = pm.KineticsRun(hpar, **par_lasers)
    assert hasattr(k_lasers, 'system')

@pytest.mark.parametrize("par_laser_bad", [
    {'uvline': {'vib': '20'}, 'odepar': {'withoutUV': False,
                                         'withoutIR': True}},
    {'uvline': {'vib': '20'}, 'odepar': {'withoutUV': False,
                                         'withoutIR': False}},
    {'uvline': {'vib': '02'}, 'odepar': {'withoutUV': False,
                                         'withoutIR': False}},
    {'uvline': {'vib': '01'}, 'odepar': {'withoutUV': False,
                                         'withoutIR': True}},
    ])
def test_KineticsRun_init_toggle_laser_setup_ValueErrors(hpar, par,
                                                         par_laser_bad):
    par_lasers = deepcopy(par)
    _nestedupdate(par_lasers, par_laser_bad)
    with pytest.raises(ValueError):
        k = pm.KineticsRun(hpar, **par_lasers)

# def test_unicode_input():
#     \u1E98\u03B5\u1E2F\u0491\u1E13\u2624\u03B7\u2118\u028A\u2602

# Update nested dictionary like `par` for tests with parametrized values,
# see https://stackoverflow.com/a/3233356
def _nestedupdate(d, u):
    for k,v in u.iteritems():
        if isinstance(v, Mapping):
            r = _nestedupdate(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d
