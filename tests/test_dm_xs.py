import json, os, math
import GollumFitPy as gf

GOLDEN = os.environ.get("DM_GOLDEN_DIR", "/golden")


def test_xs_parity():
    rows = json.load(open(os.path.join(GOLDEN, "dm_xs_golden.json")))
    nchecked = 0
    for r in rows:
        if r["sigma"] != 0.0:
            got = gf.dm_sigma(r["interaction"], r["Ei"], r["g"], r["mphi"], r["mx"])
            exp = r["sigma"]
            assert math.isclose(got, exp, rel_tol=1e-6, abs_tol=1e-300), \
                f"sigma {r['interaction']} Ei={r['Ei']}: {got} vs {exp}"
            nchecked += 1
        if r["dsigmade"] != 0.0:
            got = gf.dm_dsigmade(r["interaction"], r["Ei"], r["Ef"], r["g"], r["mphi"], r["mx"])
            exp = r["dsigmade"]
            assert math.isclose(got, exp, rel_tol=1e-6, abs_tol=1e-300), \
                f"dxs {r['interaction']} Ei={r['Ei']} Ef={r['Ef']}: {got} vs {exp}"
            nchecked += 1
    assert nchecked > 0
