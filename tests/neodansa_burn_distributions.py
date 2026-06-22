"""Burn-sample data/MC diagnostics: overlay the observed burn data with the model in energy,
cos(zenith), and RA, at (a) nuisances at their CENTER (nominal) and (b) the BEST-FIT nuisances,
plus a panel of the best-fit nuisance values. Reveals why a nuisance (e.g. galactic norm) runs away.
"""
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neodansa_likelihood as L

OUT = os.environ.get("OUT", "/out")
NBINS = (10, 10, 10)
os.makedirs(OUT, exist_ok=True)
E_edges = np.logspace(3, 6, 11); Ec = np.sqrt(E_edges[:-1] * E_edges[1:])
COSZ_c = 0.5 * (np.linspace(-1, 1, 11)[:-1] + np.linspace(-1, 1, 11)[1:])
RA_c = 0.5 * (np.linspace(0, 2 * np.pi, 11)[:-1] + np.linspace(0, 2 * np.pi, 11)[1:])
DAEMON16 = L.DAEMON16


def proj(flat):
    h = np.asarray(flat).reshape(NBINS)              # [ra, cosz, E]
    return h.sum(axis=(0, 1)), h.sum(axis=(0, 2)), h.sum(axis=(1, 2))  # E, cosz, RA


def main():
    G = L.load_model("/golden", NBINS)
    with h5py.File("/golden/neodansa_data_burn.h5") as f:
        data = f["counts"][:].astype(float)
    bf = np.load("/golden/burn_dnll.npz")
    gb, mpb, mxb = bf["best"]; x = bf["best_nuis"]; dgb = float(bf["best_dg"])
    NA, NG, conv, muon = x[0], x[1], x[2], x[19]; theta = x[3:19]
    print(f"best fit g={gb:.3e} mphi={mpb:.3e} mx={mxb:.3e} | NA={NA:.3f} dg={dgb:.3f} NG={NG:.3f} conv={conv:.3f} muon={muon:.3f}")

    # nominal: all nuisances central (norms=1, dGamma=0, theta=0), no DM (g=0)
    mu_nom = np.asarray(G.GetExpectationComponent(L.base_fp(0.0, 1.0e8, 1.0e3, 0.0), 0)).ravel()
    # best fit: fitted nuisances at the best-fit DM point
    fpb = L.base_fp(gb, mpb, mxb, dgb)
    fpb.astroNorm = NA; fpb.normGalactic = NG; fpb.convNorm = conv; fpb.muonNorm = muon
    for p, t in zip(DAEMON16, theta):
        setattr(fpb, p, float(t))
    mu_bf = np.asarray(G.GetExpectationComponent(fpb, 0)).ravel()
    print(f"totals: data={data.sum():.0f} nominal={mu_nom.sum():.0f} best-fit={mu_bf.sum():.0f}")

    dE, dC, dR = proj(data); nE, nC, nR = proj(mu_nom); bE, bC, bR = proj(mu_bf)
    # ---- distributions: 3 observables ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (xc, dd, nn, bb, xl, xlog) in zip(axes, [
            (Ec, dE, nE, bE, "reco energy [GeV]", True),
            (COSZ_c, dC, nC, bC, r"$\cos(\mathrm{zenith})$", False),
            (RA_c, dR, nR, bR, "reco RA [rad]", False)]):
        ax.errorbar(xc, dd, yerr=np.sqrt(dd), fmt="o", color="k", ms=4, label="burn data", zorder=5)
        ax.step(xc, nn, where="mid", color="#4477AA", lw=2, label="model (nominal nuis.)")
        ax.step(xc, bb, where="mid", color="#EE6677", lw=2, label="model (best-fit nuis.)")
        if xlog:
            ax.set_xscale("log")
        ax.set_xlabel(xl); ax.set_ylabel("events / bin"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Burn sample: data vs model  (best fit g={gb:.2e}, NG={NG:.1f}, conv={conv:.2f}, muon={muon:.2f})")
    fig.tight_layout(); fig.savefig(f"{OUT}/burn_data_vs_model.png", dpi=130)
    print("wrote burn_data_vs_model.png")

    # ---- nuisance pulls / values ----
    fig, ax = plt.subplots(figsize=(11, 5))
    names = ["astroNorm", "astroDGamma", "NG", "conv", "muon"] + [p.replace("hadronic", "").replace("cosmicRay", "GSF") for p in DAEMON16]
    vals = [NA, dgb, NG, conv, muon] + list(theta)
    centers = [1, 0, 1, 1, 1] + [0] * 16
    y = np.arange(len(names))
    ax.axvline(0, color="gray", lw=0.8)
    ax.errorbar(np.array(vals) - np.array(centers), y, fmt="s", color="#EE6677")
    # prior bands (relative to center): conv 0.1, muon 0.5, theta 1; NA/dg/NG free
    bands = {"conv": 0.1, "muon": 0.5}
    for i, n in enumerate(names):
        w = 1.0 if n.startswith(("HE", "VHE", "GSF")) else bands.get(n, None)
        if w:
            ax.barh(y[i], 2 * w, left=-w, height=0.6, color="0.85", zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("best-fit value $-$ center  (gray = prior $\\pm1\\sigma$; NA/dGamma/NG unconstrained)")
    ax.set_title("Burn-sample best-fit nuisance parameters"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(f"{OUT}/burn_nuisances.png", dpi=130)
    print("wrote burn_nuisances.png")


if __name__ == "__main__":
    main()
