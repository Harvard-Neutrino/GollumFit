import os
import numpy as np
import neodansa_likelihood as L

GOLDEN = os.environ.get("DM_GOLDEN_DIR", "/golden")
NBINS = (10, 10, 10)
G_, MPHI, MX = 0.316, 0.264e9, 2.07e-6 * 1e9
INJ = dict(NA=1.66, NG=2.18, conv=1.0, muon=1.0)  # injected truth


def test_injection_recovery():
    G = L.load_model(GOLDEN, NBINS, interaction="fermscal", gammaAstro=2.53)
    T = L.unit_templates(G, G_, MPHI, MX)
    data = L.model_mu(T, INJ["NA"], INJ["NG"], INJ["conv"], INJ["muon"])  # Asimov
    x, _ = L.fit_norms(T, data, x0=(1.0, 1.0, 0.5, 0.5))
    NA, NG, conv, muon = x
    # NA/NG/conv tightly constrained; muon may be weaker (Corsika template is sparse, 91 ev)
    for got, exp, name, tol in ((NA, INJ["NA"], "NA", 1e-3),
                                (NG, INJ["NG"], "NG", 1e-3),
                                (conv, INJ["conv"], "conv", 1e-3),
                                (muon, INJ["muon"], "muon", 1e-2)):
        assert abs(got - exp) / exp <= tol, f"{name}: {got} vs {exp}"


def test_profile_gamma_minimum():
    G = L.load_model(GOLDEN, NBINS, interaction="fermscal", gammaAstro=2.53)
    T = L.unit_templates(G, G_, MPHI, MX)
    data = L.model_mu(T, INJ["NA"], INJ["NG"], INJ["conv"], INJ["muon"])
    gammas = np.linspace(2.3, 2.8, 11)  # spacing 0.05; nearest grid point to 2.53 is 2.55
    prof = L.profile_gamma(GOLDEN, NBINS, G_, MPHI, MX, gammas, data)
    gmin = min(prof, key=lambda t: t[1])[0]
    assert abs(gmin - 2.53) <= (gammas[1] - gammas[0]) + 1e-9, f"gamma min at {gmin}: {prof}"


def test_profile_g_minimum():
    G = L.load_model(GOLDEN, NBINS, interaction="fermscal", gammaAstro=2.53)
    T = L.unit_templates(G, G_, MPHI, MX)
    data = L.model_mu(T, INJ["NA"], INJ["NG"], INJ["conv"], INJ["muon"])
    gs = np.array([0.1, 0.2, 0.316, 0.45, 0.6])
    prof = L.profile_g(GOLDEN, NBINS, gs, MPHI, MX, 2.53, data)
    gmin = min(prof, key=lambda t: t[1])[0]
    assert gmin == 0.316, f"g min at {gmin}: {prof}"
