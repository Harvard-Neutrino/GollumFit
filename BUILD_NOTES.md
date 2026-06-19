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
