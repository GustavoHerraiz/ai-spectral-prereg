"""Tabla conjunta post hoc por familia: formato, tamano, bytes/pixel y features v1 (C0) de varias ejecuciones de la sonda.
Uso: python3 v1-pipeline-probe-overview.py --run-dir DIR [--run-dir DIR2 ...]"""
import argparse, importlib.util, logging, os, sys
import numpy as np

LOG = logging.getLogger("overview")
spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1-pipeline-probe-analyze.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)


def q(x):
    return " ".join(f"{v:.3f}" for v in np.percentile(x, [10, 50, 90]))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", action="append", required=True); p.add_argument("--real-by-device", action="store_true"); a = p.parse_args()
    for d in a.run_dir:
        records, _ = an.load(d)
        LOG.info("\n=== %s ===", os.path.basename(d.rstrip("/")))
        LOG.info("%-22s %4s %-6s %8s %9s | %-26s | %-26s", "grupo", "n", "fmt", "MP med", "B/px med", "jpegPhaseRange p10 p50 p90", "phaseCoherence p10 p50 p90")
        if a.real_by_device:
            for r in records:
                if r["label"] == "real":
                    r["group"] = "real/" + r["rel"].split("/")[0][:16]
        groups = sorted({r["group"] for r in records}, key=lambda g: (not g.startswith("real"), g))
        for g in groups:
            rs = [r for r in records if r["group"] == g and "C0" in r["arms"]]
            fmts = ",".join(sorted({str(r.get("format")) for r in rs}))
            mp = np.median([r["width"] * r["height"] for r in rs]) / 1e6
            bpp = np.median([r["bytes"] / (r["width"] * r["height"]) for r in rs])
            f0 = [r["arms"]["C0"]["f"][0] for r in rs]; f1 = [r["arms"]["C0"]["f"][1] for r in rs]
            LOG.info("%-22s %4d %-6s %8.2f %9.3f | %-26s | %-26s", g, len(rs), fmts, mp, bpp, q(f0), q(f1))


if __name__ == "__main__":
    main()
