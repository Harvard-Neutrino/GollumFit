import os
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
    sp.neodansaExposure = 304100756.4628376 * 0.1  # burn sample (oracle used burn_sample=True)
    return sp


def _marginalize_to_oracle(hist):
    # GetExpectation returns marray (nRA, nCosth, nE). Oracle is (nE, nDec) sin_dec-ascending.
    # RA-marginalize (axis 0) -> (nCosth, nE); transpose -> (nE, nCosth);
    # cos(zenith) = -sin(dec) so reverse the costh axis to get sin(dec)-ascending.
    arr = np.asarray(hist)
    return arr.sum(axis=0).T[:, ::-1]


def test_signal_match():
    tgt = np.load(os.path.join(GOLDEN, "oracle_components.npz"))
    nbins = tuple(int(x) for x in tgt["nbins"])
    g, mphi, mx, NA, GA, NG = tgt["point"]
    G = gf.GollumFit(gf.DataPaths(), _steer(nbins))
    G.LoadNeoDANSAMC(os.path.join(GOLDEN, "neodansa_mc.h5"))
    fp = gf.FitParameters()
    fp.g, fp.mphi, fp.mx = g, mphi, mx
    fp.astroNorm, fp.normGalactic = NA, NG
    for comp, name in ((1, "astro"), (2, "galactic")):
        got = _marginalize_to_oracle(G.GetExpectationComponent(fp, comp))
        exp = tgt[name]
        assert got.shape == exp.shape, f"{name}: {got.shape} vs {exp.shape}"
        m = exp > 0
        rel = np.abs(got[m] - exp[m]) / np.abs(exp[m])
        assert rel.max() <= 1e-4, f"{name} max rel {rel.max():.2e}"
