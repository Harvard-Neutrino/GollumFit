"""NeoDANSA: round-trip tests for the 5 new (inert) fit parameters."""

import GollumFitPy as gf


def test_new_params_roundtrip():
    fp = gf.FitParameters()
    fp.g = 0.316
    fp.mphi = 0.264e9
    fp.mx = 2.07e-6 * 1e9
    fp.normGalactic = 2.18
    fp.muonNorm = 1.0
    assert fp.g == 0.316
    assert fp.normGalactic == 2.18
    assert fp.muonNorm == 1.0


def test_flag_bound_priors_have_new_fields():
    flag = gf.FitParametersFlag()
    flag.g = True
    bound = gf.FitParametersBound()
    bound.gMin = 0.0
    bound.gMax = 10.0
    pri = gf.Priors()
    pri.gCenter = 0.0
    pri.gWidth = 1e6
    assert flag.g is True
    assert bound.gMax == 10.0
    assert pri.gWidth == 1e6
