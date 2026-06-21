# GollumFit Build Notes (NeoDANSA `neodansa` branch)

Working build recipe for the NeoDANSA fork, established 2026-06-19.

## Platform
- **Docker** (Colima on macOS, Apple Silicon / linux-arm64, 4 CPU / 8 GB), base image `ubuntu:20.04` (Python 3.8), built natively for arm64.
- Image tag: `gollumfit:neodansa` (~2.67 GB).

## Build
From the repository root (build context = repo root):
```bash
docker build -f docker/Dockerfile -t gollumfit:neodansa .
```
The build compiles the full stack: SQuIDS → nuSQuIDS → PhysTools → photospline → LeptonWeighter → Diver → libGollumFit → the `GollumFitPy` pybind11 extension, and ends with an in-container `python3 -c "import GollumFitPy"` gate.

Verify at runtime:
```bash
docker run --rm gollumfit:neodansa python3 -c "import GollumFitPy as gf; print('OK', gf.FitParameters, gf.GollumFit)"
```

## Fixes applied to the upstream sources (all on the `neodansa` branch)

The upstream `docker/Dockerfile` and build did not work out of the box on this layout/arch. Four fixes:

1. **`docker/Dockerfile` — rewritten for the current layout.** Upstream referenced the old `Fit/` + `sources/` directories; the current repo uses `src/` + `include/` + `python/` + `vendor/` submodules + a root `Makefile`/`configure`. The Dockerfile now:
   - builds with the **repo root as context** (not `docker/`);
   - **builds vendored deps from `vendor/` FIRST**, then copies the GollumFit source (`src/`, `include/`, `python/`, `Makefile`, `configure`) — so editing GollumFit source does NOT trigger a full dependency rebuild;
   - drops the Jupyter `notebook` pip install (pulls `jupyterlab>=4.4.9` which needs Python ≥3.9; base is 3.8);
   - adds `pybind11` to the pip install (the bindings build needs it) and `gfortran` (Diver is Fortran).

2. **HDF5 linker path.** On Ubuntu 20.04 the HDF5 libs live in `/usr/lib/<arch>-linux-gnu/hdf5/serial/`, which sub-builds' Makefiles don't add to the link path → `cannot find -lhdf5`. The Dockerfile symlinks `libhdf5.so`/`libhdf5_hl.so` into `/usr/lib`.

3. **Diver.** The Python bindings `#include "GollumFitDiverWrapper.h"` and link `-ldiver` unconditionally (despite docs calling Diver "optional"). The Dockerfile clones `diveropt/Diver`, builds `libdiver.so`, and installs `libdiver.so` + `diver.hpp` to `/usr`.

4. **Link order (`configure:822` and `Makefile:105`) — the key fix.** The dynamic-link recipe listed libraries before objects:
   `$(CXX) $(DYN_OPT) $(LDFLAGS) -o $(DYN_PRODUCT) $(OBJECTS)`.
   On Linux (default `--as-needed`) this drops every shared dependency, so `libGollumFit.so` had no `NEEDED` entry for `libPhysTools.so` → runtime `ImportError: undefined symbol: phys_tools::hdf_interface::DatatypeRegistry`. (It worked on macOS because Darwin's linker isn't `--as-needed`.) Fixed to objects-before-libraries:
   `$(CXX) $(DYN_OPT) -o $(DYN_PRODUCT) $(OBJECTS) $(LDFLAGS)`.
   Note `./configure` **regenerates** the `Makefile`, so the fix lives in `configure` (line ~822); the committed `Makefile` is patched too for the no-configure path.

## Running examples
The shipped `examples/expectations/generate_expectation.py` loads all splines and constructs the `GollumFit` object successfully, but needs a FastMC file (`../FastMC/example.fastmc`) which must be generated from real MC. `monte_carlo/` ships only LeptonWeighter `.lic` files, not MC events. Bind-mount real MC and run `examples/FastMC/generate_fastMC.py` first, or use NeoDANSA's own converted DANSA MC (later plan). Run examples with the repo bind-mounted so `resources/` is visible:
```bash
docker run --rm -v "$PWD":/work -w /work/examples/expectations gollumfit:neodansa python3 generate_expectation.py LABEL
```

## NeoDANSA native weighter (2026-06-20)

NeoDANSA now uses GollumFit's **native** weighter machinery end-to-end (no bespoke
flux math):
- **astro** = `astroNorm · cachedAstroWeight · brokenpowerlawTiltWeighter(astroPivot, astroDeltaGamma, astroDeltaGammaSec) · DM_att`. `cachedAstroWeight` is built in `LoadNeoDANSAMC` from `examples/fluxes/astro.hdf5` (oscillation-averaged) × per-event `OneWeight/(NEvents·nFiles)` × exposure. DANSA single-PL = equal slopes; the index floats via `astroDeltaGamma`.
- **conv** = `convNorm · ConvFluxWeigther(16 DAEMONFlux params)`. The 16 gradient caches are built in `LoadNeoDANSABackground` from GollumFit's shipped nuSQuIDS gradient tables (`he_K+_.hdf5 … GSF_6_.hdf5`) — GollumFit computes the gradients.
- **DM attenuation** is the only bespoke per-event factor.
- MC re-exported with `pdg` (flavor, recovered from snowstorm via the unique 5-key `run,event,subevent,trueE,true_zen`), `trueZenith`, and `OneWeight`.
- Flux dir passed via `steeringParams.neodansaFluxDir`; tables mounted at runtime (`-v examples/fluxes:/fluxes`, 853 MB).

### Validated
conv total = 2509.6 (matches DANSA MCEq exactly); DAEMONFlux 1σ shifts ~few %; Asimov
inject/recover of all non-detector nuisances (astroNorm, astro index, NG, conv, 16
DAEMONFlux w/ unit priors, muon) closes. `tests/neodansa_native_validate.py` makes the
figures. **Histogram flatten order is `[ra, cos(zenith), energy]`** — energy is the inner
axis; energy projection = `reshape(10,10,10).sum(axis=(0,1))`.

### KNOWN LIMITATIONS — flux-table recompute required before physics results
The shipped `examples/fluxes/*.hdf5` (built by `resources/FluxOscCalculator/*_with_interactions*.cpp`) have two problems for NuGen-based MC:
1. **Earth absorption double-counted.** The tables include the nuSQuIDS interaction/attenuation term, and NuGen MC already accounts for Earth absorption (via `OneWeight`/propagation). Recompute the flux tables with **interactions shut OFF** (oscillations only, surface flux) so the absorption comes solely from the MC. Use `resources/FluxOscCalculator` (and the astro/conv/prompt calculators) with the interaction term disabled.
2. **Zenith range too small.** Tables cover `cos(theta) ∈ [-1, 0.2]` only; the loader currently clamps down-going events to the horizon. Recompute over the full `[-1, 1]`.

Also: the 16 DAEMONFlux gradient tables / errors should be regenerated from the DAEMONFlux
app + `resources/FluxOscCalculator/errors_ddm_flux_calculator_with_interactions_from_file.cpp`
(interactions off). Once the corrected tables exist, the conv **nominal** can also become
fully native (it currently uses DANSA's validated MCEq weight because the atmo pickle's
`OneWeight` convention is opaque — the 16 gradient *shapes* are already native via the table ratio).

## osc-only daemonflux flux tables (2026-06-21) — addresses the flux-recompute TODO

`resources/FluxOscCalculator/osc_only_atmospheric_flux_calculator.cpp`: propagates a surface
flux with **oscillations only** (`nuSQUIDSAtm<>(linspace(-1,1,100), logspace(1e2,1e7,350), 3,
both, /*iinteraction=*/false)` + EarthAtm for the MSW matter potential) over the **full zenith
range**, writing a nuSQuIDS HDF5. No Earth absorption (NuGen already accounts for it).

Pipeline:
1. `NeoDANSA/scripts/make_daemonflux_dat.py` (run in the DANSA env; uses the daemonflux
   library + its cache) dumps the conventional surface flux (nominal + 16 DAEMONFlux gradients)
   to 8-col .dat on the nuSQuIDS grid. daemonflux returns E^3*dPhi/dE -> divide by E^3.
2. Build object-FIRST (libs-before-object link order silently produces a broken binary, and
   the macOS bind mount strips +x — compile into container-local /tmp):
   `g++ -O3 -std=c++11 -pthread $(pkg-config nusquids --cflags) -I/usr/include/hdf5/serial
    osc_only_atmospheric_flux_calculator.cpp $(pkg-config nusquids --libs) -lhdf5 -lhdf5_hl -o /tmp/calc`
3. Run on each .dat -> `<name>.hdf5` (atmospheric.hdf5, he_K+_.hdf5 … GSF_6_.hdf5) + an
   isotropic-E^-2 `astro.hdf5`. Point `neodansaFluxDir` there.

Loader (`neodansa_evalFlux`) now spans full cos(theta) [-1,1] and E [1e2,1e7] (no horizon clamp).
Validated: full zenith works (no out-of-bounds), conv = 2510, DAEMONFlux 1σ gradients physical
(GSF1 +4.5%, K+2P +9.8%); inject/recover closes to <1e-3.

**Atmo absolute normalization (known):** DANSA's atmo weight uses precomputed
`weights_MCEq` (= flux × oneweight × livetime) directly and the reduced atmo pickle dropped its
generation `NEvents`/`nFiles`, so an absolute `daemonflux × oneweight` conv nominal is NOT
recoverable (the snowstorm `NEvents`/`nFiles` belong to a different production → off by ~1e7).
So conv NOMINAL stays DANSA MCEq (= the H3a+Sibyll conv flux, i.e. the DANSA oracle by
construction) and the daemonflux osc-only tables supply the gradient SHAPES (scale cancels).
Fully-native conv would need the atmo MC's own generation normalization, or reweighting the
signal NuGen (which keeps correct OneWeight) to conv flux + a self-veto — follow-up.
