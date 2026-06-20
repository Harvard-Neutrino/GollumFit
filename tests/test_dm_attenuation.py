import json, os
import GollumFitPy as gf

GOLDEN = os.environ.get("DM_GOLDEN_DIR", "/golden")


def test_att_parity():
    rows = json.load(open(os.path.join(GOLDEN, "dm_att_golden.json")))
    for r in rows:
        got = gf.dm_attenuation(r["interaction"], r["g"], r["mphi"], r["mx"],
                                r["gamma"], r["columnDens"], r["trueE_gev"])
        exp = r["att"]
        assert len(got) == len(exp)
        for a, b in zip(got, exp):
            rel = abs(a - b) / (abs(b) + 1e-300)
            assert rel <= 1e-5, f"{r['interaction']}: {a} vs {b} (rel {rel})"
