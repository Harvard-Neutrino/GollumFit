import os
import h5py
import numpy as np
import GollumFitPy as gf

GOLDEN = os.environ.get("DM_GOLDEN_DIR", "/golden")


def _steer(nbins):
    sp = gf.SteeringParams()
    sp.minFitEnergy, sp.maxFitEnergy = 1e3, 1e6
    sp.logEbinEdge, sp.logEbinWidth = np.log10(1e3), (np.log10(1e6) - np.log10(1e3)) / nbins[0]
    sp.minCosth, sp.maxCosth, sp.cosThbinEdge, sp.cosThbinWidth = -1.0, 1.0, -1.0, 2.0 / nbins[1]
    sp.minRA, sp.maxRA, sp.raBinEdge, sp.raBinWidth = 0.0, 2 * np.pi, 0.0, 2 * np.pi / nbins[2]
    sp.useNeoDANSAWeighter = True
    sp.interaction = "fermscal"
    sp.gammaAstro = 2.53
    sp.gammaGalactic = 2.7
    sp.neodansaExposure = 304100756.4628376 * 0.1
    return sp


def _marg(hist):
    # (nRA, nCosth, nE) -> (nE, nDec) sin_dec-ascending (cos(zenith) = -sin(dec))
    arr = np.asarray(hist)
    return arr.sum(axis=0).T[:, ::-1]


def _load(nbins):
    G = gf.GollumFit(gf.DataPaths(), _steer(nbins))
    G.LoadNeoDANSAMC(os.path.join(GOLDEN, "neodansa_mc.h5"))
    G.LoadNeoDANSABackground(os.path.join(GOLDEN, "neodansa_atmo.h5"),
                             os.path.join(GOLDEN, "neodansa_muon.h5"))
    return G


def test_background_loads():
    tgt = np.load(os.path.join(GOLDEN, "oracle_components.npz"))
    nbins = tuple(int(x) for x in tgt["nbins"])
    n_sig = h5py.File(os.path.join(GOLDEN, "neodansa_mc.h5"))["recoEnergy"].shape[0]
    n_atm = h5py.File(os.path.join(GOLDEN, "neodansa_atmo.h5"))["recoEnergy"].shape[0]
    n_mu = h5py.File(os.path.join(GOLDEN, "neodansa_muon.h5"))["recoEnergy"].shape[0]
    G = _load(nbins)
    assert G.NumMCEvents() == n_sig + n_atm + n_mu


def test_total_match():
    tgt = np.load(os.path.join(GOLDEN, "oracle_components.npz"))
    nbins = tuple(int(x) for x in tgt["nbins"])
    g, mphi, mx, NA, GA, NG = tgt["point"]
    G = _load(nbins)
    fp = gf.FitParameters()
    fp.g, fp.mphi, fp.mx = g, mphi, mx
    fp.astroNorm, fp.normGalactic = NA, NG
    fp.convNorm, fp.muonNorm = 1.0, 1.0
    atmo = _marg(G.GetExpectationComponent(fp, 3))
    muon = _marg(G.GetExpectationComponent(fp, 4))
    bkg = atmo + muon
    total = _marg(G.GetExpectationComponent(fp, 0))
    for got, name in ((bkg, "background"), (total, "total")):
        exp = tgt[name]
        assert got.shape == exp.shape, f"{name}: {got.shape} vs {exp.shape}"
        m = exp > 0
        rel = np.abs(got[m] - exp[m]) / np.abs(exp[m])
        assert rel.max() <= 1e-4, f"{name} max rel {rel.max():.2e}"
