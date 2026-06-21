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


FLUX_DIR = os.environ.get("DM_FLUX_DIR", "/fluxes")  # GollumFit's shipped nuSQuIDS flux tables


def make_steer(nbins, interaction="fermscal", gammaAstro=2.53, gammaGalactic=2.7,
               exposure=BURN_EXPOSURE, flux_dir=None):
    sp = gf.SteeringParams()
    sp.minFitEnergy, sp.maxFitEnergy = 1e3, 1e6
    sp.logEbinEdge, sp.logEbinWidth = np.log10(1e3), (np.log10(1e6) - np.log10(1e3)) / nbins[0]
    sp.minCosth, sp.maxCosth, sp.cosThbinEdge, sp.cosThbinWidth = -1.0, 1.0, -1.0, 2.0 / nbins[1]
    sp.minRA, sp.maxRA, sp.raBinEdge, sp.raBinWidth = 0.0, 2 * np.pi, 0.0, 2 * np.pi / nbins[2]
    sp.useNeoDANSAWeighter = True
    sp.interaction = interaction
    sp.gammaAstro, sp.gammaGalactic = gammaAstro, gammaGalactic
    sp.neodansaExposure = exposure
    sp.neodansaFluxDir = flux_dir or FLUX_DIR
    return sp


def load_model(golden_dir, nbins, interaction="fermscal", gammaAstro=2.53,
               gammaGalactic=2.7, exposure=BURN_EXPOSURE, flux_dir=None):
    G = gf.GollumFit(gf.DataPaths(), make_steer(nbins, interaction, gammaAstro,
                                                gammaGalactic, exposure, flux_dir))
    G.LoadNeoDANSAMC(os.path.join(golden_dir, "neodansa_mc.h5"))
    G.LoadNeoDANSABackground(os.path.join(golden_dir, "neodansa_atmo.h5"),
                             os.path.join(golden_dir, "neodansa_muon.h5"))
    return G


DAEMON16 = ("hadronicHEkp", "hadronicHEkm", "hadronicVHE1pip", "hadronicVHE1pim",
            "hadronicVHE3kp", "hadronicVHE3km", "hadronicVHE3pip", "hadronicVHE3pim",
            "hadronicVHE3p", "hadronicVHE3n", "cosmicRay1", "cosmicRay2", "cosmicRay3",
            "cosmicRay4", "cosmicRay5", "cosmicRay6")


def base_fp(g, mphi, mx, astroDeltaGamma=0.0):
    """FitParameters at nominal nuisances (all DAEMONFlux = 0, unit norms)."""
    fp = gf.FitParameters()
    fp.g, fp.mphi, fp.mx = g, mphi, mx
    fp.astroNorm = 1.0; fp.normGalactic = 1.0; fp.convNorm = 1.0; fp.muonNorm = 1.0
    fp.astroPivot = 5.0  # log10(100 TeV / GeV)
    fp.astroDeltaGamma = astroDeltaGamma; fp.astroDeltaGammaSec = astroDeltaGamma
    for p in DAEMON16:
        setattr(fp, p, 0.0)  # nominal; nonzero garbage -> ConvFluxWeigther FLT_MAX on negative flux
    return fp


def comp(G, fp, c):
    return np.asarray(G.GetExpectationComponent(fp, c))


def astro_template(G, g, mphi, mx, dgamma):
    """Astro component (astroNorm=1) at a given index tilt dgamma (native broken-PL tilt)."""
    return comp(G, base_fp(g, mphi, mx, dgamma), 1)


def static_templates(G, g, mphi, mx):
    """dgamma-independent templates: galactic, muon, conv nominal + 16 conv gradient shapes.

    conv(theta) = conv0 + sum_k theta_k * conv_grad_k  (linear, native ConvFluxWeigther).
    """
    fp0 = base_fp(g, mphi, mx)
    T = {"galactic": comp(G, fp0, 2), "muon": comp(G, fp0, 4), "conv0": comp(G, fp0, 3)}
    grads = []
    for p in DAEMON16:
        fpk = base_fp(g, mphi, mx); setattr(fpk, p, 1.0)
        grads.append(comp(G, fpk, 3) - T["conv0"])
    T["conv_grads"] = np.array(grads)  # (16, nbins)
    return T


def model_mu(Tastro, S, NA, NG, conv, theta, muon):
    """Total expectation: NA*astro + NG*gal + conv*(conv0 + theta.conv_grads) + muon*muon."""
    conv_shape = S["conv0"] + np.tensordot(theta, S["conv_grads"], axes=(0, 0))
    return NA * Tastro + NG * S["galactic"] + conv * conv_shape + muon * S["muon"]


def poisson_nll(mu, data):
    mu = np.clip(mu, 1e-300, None)
    return float(np.sum(mu - data * np.log(mu)))


def fit_all_nuisances(G, g, mphi, mx, data, x0=None):
    """Joint fit over ALL non-detector nuisances (native model):
    x = [astroNorm, astroDeltaGamma, NG, conv, theta_1..16, muon] (20 params).

    Unit Gaussian priors on the 16 DAEMONFlux nuisances (GollumFit convention). The astro
    template is recomputed (cached) only when astroDeltaGamma changes; everything else is
    linear in precomputed templates.
    """
    from scipy.optimize import minimize
    S = static_templates(G, g, mphi, mx)
    cache = {"dg": None, "T": None}

    def astro_T(dg):
        if cache["dg"] != dg:
            cache["T"] = astro_template(G, g, mphi, mx, dg); cache["dg"] = dg
        return cache["T"]

    def unpack(x):
        return x[0], x[1], x[2], x[3], np.asarray(x[4:20]), x[20]

    def nll(x):
        NA, dg, NG, conv, theta, muon = unpack(x)
        mu = model_mu(astro_T(dg), S, NA, NG, conv, theta, muon)
        return poisson_nll(mu, data) + 0.5 * float(np.dot(theta, theta))  # unit priors on the 16

    if x0 is None:
        x0 = [1.0, 0.0, 1.0, 1.0] + [0.0] * 16 + [1.0]
    bounds = [(0, None), (-1.0, 1.0), (0, None), (0, None)] + [(-5, 5)] * 16 + [(0, None)]
    res = minimize(nll, np.array(x0, float), method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 5000, "maxfun": 50000, "ftol": 1e-13, "gtol": 1e-10})
    NA, dg, NG, conv, theta, muon = unpack(res.x)
    return dict(astroNorm=NA, astroDeltaGamma=dg, normGalactic=NG, convNorm=conv,
                theta=theta, muonNorm=muon, nll=res.fun), res


def profile_g(golden_dir, nbins, gs, mphi, mx, dgamma, data):
    """Profile NA/NG/conv/muon (norms only, fixed nuisances) vs DM coupling g."""
    from scipy.optimize import minimize
    out = []
    for g in gs:
        G = load_model(golden_dir, nbins, "fermscal")
        S = static_templates(G, g, mphi, mx); Ta = astro_template(G, g, mphi, mx, dgamma)
        def nll(x):
            return poisson_nll(model_mu(Ta, S, x[0], x[1], x[2], np.zeros(16), x[3]), data)
        r = minimize(nll, [1, 1, 1, 1], method="L-BFGS-B", bounds=[(0, None)] * 4)
        out.append((float(g), r.fun))
    return out
