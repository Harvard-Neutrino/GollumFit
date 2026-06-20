"""NeoDANSA forward-folding likelihood helpers (M6 injection-recovery).

GollumFit's native MinLLH cannot be used here (custom-loaded events don't populate
the stock WeighterMaker caches, and its autodiff can't pass through the GSL cascade
eigensolve), so we use GetExpectationComponent as the forward model. The expectation
is linear in the four normalizations, so we build unit-norm component templates once
and minimize a Poisson NLL with SciPy. Detector systematics are not in the model.
"""
import os
import numpy as np
import GollumFitPy as gf

BURN_EXPOSURE = 304100756.4628376 * 0.1


def make_steer(nbins, interaction="fermscal", gammaAstro=2.53, gammaGalactic=2.7,
               exposure=BURN_EXPOSURE):
    sp = gf.SteeringParams()
    sp.minFitEnergy, sp.maxFitEnergy = 1e3, 1e6
    sp.logEbinEdge, sp.logEbinWidth = np.log10(1e3), (np.log10(1e6) - np.log10(1e3)) / nbins[0]
    sp.minCosth, sp.maxCosth, sp.cosThbinEdge, sp.cosThbinWidth = -1.0, 1.0, -1.0, 2.0 / nbins[1]
    sp.minRA, sp.maxRA, sp.raBinEdge, sp.raBinWidth = 0.0, 2 * np.pi, 0.0, 2 * np.pi / nbins[2]
    sp.useNeoDANSAWeighter = True
    sp.interaction = interaction
    sp.gammaAstro, sp.gammaGalactic = gammaAstro, gammaGalactic
    sp.neodansaExposure = exposure
    return sp


def load_model(golden_dir, nbins, interaction="fermscal", gammaAstro=2.53,
               gammaGalactic=2.7, exposure=BURN_EXPOSURE):
    G = gf.GollumFit(gf.DataPaths(), make_steer(nbins, interaction, gammaAstro,
                                                gammaGalactic, exposure))
    G.LoadNeoDANSAMC(os.path.join(golden_dir, "neodansa_mc.h5"))
    G.LoadNeoDANSABackground(os.path.join(golden_dir, "neodansa_atmo.h5"),
                             os.path.join(golden_dir, "neodansa_muon.h5"))
    return G


def unit_templates(G, g, mphi, mx):
    """Unit-norm 3D component templates {astro,galactic,atmo,muon}."""
    fp = gf.FitParameters()
    fp.g, fp.mphi, fp.mx = g, mphi, mx
    fp.astroNorm = 1.0; fp.normGalactic = 1.0; fp.convNorm = 1.0; fp.muonNorm = 1.0
    return {
        "astro": np.asarray(G.GetExpectationComponent(fp, 1)),
        "galactic": np.asarray(G.GetExpectationComponent(fp, 2)),
        "atmo": np.asarray(G.GetExpectationComponent(fp, 3)),
        "muon": np.asarray(G.GetExpectationComponent(fp, 4)),
    }


def model_mu(templates, NA, NG, conv, muon):
    return (NA * templates["astro"] + NG * templates["galactic"]
            + conv * templates["atmo"] + muon * templates["muon"])


def poisson_nll(mu, data):
    mu = np.clip(mu, 1e-300, None)
    return float(np.sum(mu - data * np.log(mu)))


def fit_norms(templates, data, x0=(1.0, 1.0, 1.0, 1.0)):
    from scipy.optimize import minimize
    def nll(x):
        return poisson_nll(model_mu(templates, *x), data)
    res = minimize(nll, np.array(x0, float), method="L-BFGS-B", bounds=[(0, None)] * 4)
    return res.x, res.fun


def profile_gamma(golden_dir, nbins, g, mphi, mx, gammas, data, gammaGalactic=2.7,
                  exposure=BURN_EXPOSURE):
    # Reload the model per gamma (SetSteeringParams is not bound in pybind11).
    out = []
    for ga in gammas:
        G = load_model(golden_dir, nbins, "fermscal", ga, gammaGalactic, exposure)
        T = unit_templates(G, g, mphi, mx)
        _, nll = fit_norms(T, data)
        out.append((float(ga), nll))
    return out


def profile_g(golden_dir, nbins, gs, mphi, mx, gammaAstro, data, exposure=BURN_EXPOSURE):
    out = []
    for g in gs:
        G = load_model(golden_dir, nbins, "fermscal", gammaAstro, 2.7, exposure)
        T = unit_templates(G, g, mphi, mx)
        _, nll = fit_norms(T, data)
        out.append((float(g), nll))
    return out
