from __future__ import division
import pytest
import popmodel as pm
from popmodel import simspec # not included in __init__
from pkg_resources import resource_filename
from numpy import isclose

def test_sigma_eff_workflow_uv_q13_line():
    q13 = simspec.sigma_tot(1/308.1541e-7, 3.5, 3.5, 5.700e5)
    a = pm.absprofile.AbsProfile(1/308.1541e-7)
    a.makeprofile(press=760., T=300., g_air=0.112)
    se = simspec.sigma_eff(0.075, q13, a.pop/a.binwidth)
    assert isclose(se.max(), 1.4468274436093546e-16, rtol=5e-4, atol=0)

def test_simline_workflow_ir_p13_line(hpar):
    ser = simspec.simline(hpar[hpar['label']==u'P_1(3)ff'], press=760)
    assert isclose(ser.max(), 2.641894669817001e-19, rtol=5e-3, atol=0)

def test_simspec_workflow(hpar):
    df = simspec.simspec(hpar, press=760)
    assert isclose(df.max().max(), 2.641894669817001e-19, rtol=5e-3, atol=0)
