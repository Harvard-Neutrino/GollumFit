"""Compare the GollumFit native-weighter (DFWM) expectation to the DANSA oracle, per component.

The DANSA oracle (golden/oracle_components.npz) holds astro/galactic/background/total on a
(nE, nDec) grid at a fixed DM point + nuisances. We compute the same with GollumFit's native
DFWM weighter and overlay the energy spectra. astro uses the E^-2 astro.hdf5 baseline tilted to
DANSA's index gamma via astroDeltaGamma = gamma - 2; conv nominal = DANSA MCEq; muon = Corsika.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neodansa_likelihood as L

OUT = os.environ.get("OUT", "/out")
NBINS = (10, 10, 10)
os.makedirs(OUT, exist_ok=True)

orc = np.load("/golden/oracle_components.npz")
g, mphi, mx, NA, GA, NG = [float(x) for x in orc["point"]]
E_edges = orc["energy_edges"]
Ec = np.sqrt(E_edges[:-1] * E_edges[1:])
print(f"oracle point: g={g} mphi={mphi} mx={mx} NA={NA} gamma={GA} NG={NG}")


def Espec(h3):                       # native 3D [ra,cosz,E] -> E spectrum
    return np.asarray(h3).reshape(NBINS).sum(axis=(0, 1))


def main():
    G = L.load_model("/golden", NBINS)
    # native expectation at the oracle point + nuisances (E^-2 baseline -> gamma via dGamma=GA-2)
    fp = L.base_fp(g, mphi, mx, astroDeltaGamma=GA - 2.0)
    fp.astroNorm = NA; fp.normGalactic = NG; fp.convNorm = 1.0; fp.muonNorm = 1.0
    nat = {n: Espec(L.comp(G, fp, c)) for n, c in (("astro", 1), ("galactic", 2),
                                                   ("conv", 3), ("muon", 4))}
    nat["background"] = nat["conv"] + nat["muon"]   # DANSA bg = atmo-nu + Corsika
    nat["total"] = nat["astro"] + nat["galactic"] + nat["background"]

    dan = {k: orc[k].sum(axis=1) for k in ("astro", "galactic", "background", "total")}

    print("\ncomponent     native    DANSA    ratio")
    for k in ("astro", "galactic", "background", "total"):
        print(f"  {k:11s} {nat[k].sum():8.2f} {dan[k].sum():8.2f}   {nat[k].sum()/dan[k].sum():.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, k in zip(axes.ravel(), ("astro", "galactic", "background", "total")):
        ax.step(Ec, nat[k], where="mid", color="#EE6677", lw=2, label="GollumFit native")
        ax.step(Ec, dan[k], where="mid", color="k", lw=1.5, ls="--", label="DANSA oracle")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-2, None)
        ax.set_xlabel("reco energy [GeV]"); ax.set_ylabel("events / bin")
        ax.set_title(f"{k}  (native/DANSA = {nat[k].sum()/dan[k].sum():.3f})")
        ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"NeoDANSA native weighter vs DANSA oracle  (g={g:.3g}, gamma={GA}, NA={NA}, NG={NG})")
    fig.tight_layout(); fig.savefig(f"{OUT}/native_vs_dansa.png", dpi=130)
    print("wrote native_vs_dansa.png")


if __name__ == "__main__":
    main()
