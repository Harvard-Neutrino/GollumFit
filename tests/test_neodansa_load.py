import os
import h5py
import numpy as np
import GollumFitPy as gf

GOLDEN = os.environ.get("DM_GOLDEN_DIR", "/golden")


def _steer(nbins=(10, 10, 10)):
    sp = gf.SteeringParams()
    sp.minFitEnergy, sp.maxFitEnergy = 1e3, 1e6
    sp.logEbinEdge, sp.logEbinWidth = np.log10(1e3), (np.log10(1e6) - np.log10(1e3)) / nbins[0]
    sp.minCosth, sp.maxCosth, sp.cosThbinEdge, sp.cosThbinWidth = -1.0, 1.0, -1.0, 2.0 / nbins[1]
    sp.minRA, sp.maxRA, sp.raBinEdge, sp.raBinWidth = 0.0, 2 * np.pi, 0.0, 2 * np.pi / nbins[2]
    sp.useNeoDANSAWeighter = True
    return sp


def test_load_counts_and_runs():
    path = os.path.join(GOLDEN, "neodansa_mc.h5")
    N = h5py.File(path)["recoEnergy"].shape[0]
    dp = gf.DataPaths()  # all spline/flux paths empty -> no flux machinery
    g = gf.GollumFit(dp, _steer())
    g.LoadNeoDANSAMC(path)
    assert g.NumMCEvents() == N, f"{g.NumMCEvents()} != {N}"
    # GetExpectation must run (base/DFWM path returns ~0 with empty caches; shape sanity only)
    fp = gf.FitParameters()
    hist = np.asarray(g.GetExpectation(fp))
    assert hist.ndim == 3, f"expected 3D histogram, got {hist.shape}"
