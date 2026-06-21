"""Native-weighter validation: component distributions + all-nuisance inject/recover.

Runs inside the GollumFit container with /golden (MC) + /fluxes (nuSQuIDS tables) mounted.
Writes figures to $OUT (default /out). All flux weighting is done by GollumFit's native
weighters; the only bespoke physics is the DM cascade attenuation (g=0 here -> null).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neodansa_likelihood as L

OUT = os.environ.get("OUT", "/out")
NBINS = (10, 10, 10)
MPHI, MX = 0.264e9, 2.07e3
G_DM = 0.0  # null (no DM attenuation) for the nuisance round trip
os.makedirs(OUT, exist_ok=True)

E_edges = np.logspace(3, 6, NBINS[0] + 1)
Ec = np.sqrt(E_edges[:-1] * E_edges[1:])


def spectrum(h3):  # histogram flattens as [ra, cos(zenith), energy]; energy is the inner axis
    return np.asarray(h3).reshape(NBINS).sum(axis=(0, 1))


def main():
    G = L.load_model("/golden", NBINS)
    fp = L.base_fp(G_DM, MPHI, MX, astroDeltaGamma=0.0)
    comps = {n: L.comp(G, fp, c) for n, c in (("astro", 1), ("galactic", 2), ("conv", 3), ("muon", 4))}
    print("component totals:", {k: round(float(v.sum()), 2) for k, v in comps.items()})

    # ---- Figure 1: energy spectrum (per-component rate, log-log) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    total = np.zeros(NBINS[0])
    for n, col in (("conv", "#4477AA"), ("muon", "#999933"), ("astro", "#EE6677"), ("galactic", "#228833")):
        s = spectrum(comps[n]); total += s
        ax.step(Ec, s, where="mid", color=col, lw=2, label=n)
    ax.step(Ec, total, where="mid", color="k", lw=1.5, ls="--", label="total")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("reco energy [GeV]"); ax.set_ylabel("events / bin")
    ax.set_ylim(1e-2, None); ax.set_title("NeoDANSA native weighter — component rates")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/native_distributions_spectrum.png", dpi=130)
    print("wrote native_distributions_spectrum.png")

    # ---- Inject/recover ALL non-detector nuisances (Asimov) ----
    truth = dict(astroNorm=1.20, astroDeltaGamma=0.15, normGalactic=1.50, convNorm=1.10, muonNorm=0.90)
    S = L.static_templates(G, G_DM, MPHI, MX)
    Ta = L.astro_template(G, G_DM, MPHI, MX, truth["astroDeltaGamma"])
    theta_truth = np.zeros(16)  # DAEMONFlux injected at nominal (priors) -> recover ~0
    data = L.model_mu(Ta, S, truth["astroNorm"], truth["normGalactic"], truth["convNorm"],
                      theta_truth, truth["muonNorm"])
    res, _ = L.fit_all_nuisances(G, G_DM, MPHI, MX, data)
    print("\ninject -> recover:")
    rows = [("astroNorm", truth["astroNorm"], res["astroNorm"]),
            ("astroDeltaGamma", truth["astroDeltaGamma"], res["astroDeltaGamma"]),
            ("normGalactic", truth["normGalactic"], res["normGalactic"]),
            ("convNorm", truth["convNorm"], res["convNorm"]),
            ("muonNorm", truth["muonNorm"], res["muonNorm"])]
    for nm, t, r in rows:
        print(f"  {nm:16s} inj={t:+.4f} rec={r:+.4f} d={r-t:+.2e}")
    print("  DAEMONFlux theta recovered |max| = %.2e (injected 0)" % np.abs(res["theta"]).max())

    # ---- Figure 2: injection recovery ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [r[0] for r in rows]; inj = [r[1] for r in rows]; rec = [r[2] for r in rows]
    y = np.arange(len(names))
    axes[0].errorbar(inj, y - 0.1, fmt="o", color="k", label="injected")
    axes[0].errorbar(rec, y + 0.1, fmt="s", color="#EE6677", label="recovered")
    axes[0].set_yticks(y); axes[0].set_yticklabels(names); axes[0].legend()
    axes[0].set_title("Norms + astro index: injected vs recovered"); axes[0].grid(alpha=0.3)
    axes[1].bar(np.arange(16), res["theta"], color="#4477AA")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(np.arange(16)); axes[1].set_xticklabels([p.replace("hadronic", "").replace("cosmicRay", "GSF") for p in L.DAEMON16], rotation=90, fontsize=7)
    axes[1].set_ylabel(r"recovered $\theta$ (injected 0, unit priors)")
    axes[1].set_title("16 DAEMONFlux nuisances"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/native_injection_recovery.png", dpi=130)
    print("wrote native_injection_recovery.png")


if __name__ == "__main__":
    main()
