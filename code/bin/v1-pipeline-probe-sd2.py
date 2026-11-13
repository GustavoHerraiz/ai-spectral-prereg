"""Exploracion post hoc de SD2 frente a SDXL (sonda v1, brazo C0). Solo lectura de features.shard*.jsonl; sin hipotesis confirmatoria.
Uso: python3 v1-pipeline-probe-sd2.py --run-dir DIR [--a stable-diffusion-2 --b stable-diffusion-xl]"""
import argparse, importlib.util, logging, os, sys, collections
import numpy as np
from scipy.stats import spearmanr

LOG = logging.getLogger("sd2")
spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1-pipeline-probe-analyze.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)


def describe(name, recs):
    sizes = collections.Counter(f"{r['width']}x{r['height']}" for r in recs)
    mult = {m: np.mean([r["width"] % m == 0 and r["height"] % m == 0 for r in recs]) for m in (8, 16, 64)}
    off = collections.Counter((r["left"] % 8, r["top"] % 8) for r in recs)
    LOG.info("%s n=%d tamanos=%s", name, len(recs), sizes.most_common(4))
    LOG.info("%s multiplo(8,16,64)=%s origen recorte mod 8=%s", name, {m: round(float(v), 2) for m, v in mult.items()}, dict(off))
    bpp = np.array([r["bytes"] / (r["width"] * r["height"]) for r in recs])
    for k, feat in enumerate(an.FEATURES):
        x = an.values(recs, "C0", k)
        rho, p = spearmanr(x, bpp)
        LOG.info("%s %-16s mediana=%.3f; rho(feature, B/px)=%.2f (p=%.3g)", name, feat, np.median(x), rho, p)


def extremes(name, recs, n=5):
    x = an.values(recs, "C0", 0)
    order = np.argsort(x)
    for label, idx in (("bajas", order[:n]), ("altas", order[-n:])):
        for i in idx:
            r = recs[i]
            LOG.info("%s %s jpegPhaseRange=%.3f %dx%d B/px=%.2f %s", name, label, x[i], r["width"], r["height"], r["bytes"] / (r["width"] * r["height"]), r["rel"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", required=True)
    p.add_argument("--a", default="stable-diffusion-2"); p.add_argument("--b", default="stable-diffusion-xl")
    a = p.parse_args()
    records, _ = an.load(a.run_dir)
    _, fams = an.split(records)
    for name in (a.a, a.b):
        if name not in fams:
            LOG.error("familia ausente: %s", name); sys.exit(1)
        describe(name, fams[name])
    extremes(a.a, fams[a.a]); extremes(a.b, fams[a.b])
    sa = fams[a.a]
    for mn in (1024, 768):
        sub = [r for r in sa if min(r["width"], r["height"]) >= mn]
        LOG.info("%s con lado menor >= %d: n=%d", a.a, mn, len(sub))
        if len(sub) >= 5:
            LOG.info("   mediana jpegPhaseRange=%.3f", np.median(an.values(sub, "C0", 0)))
    sub = [r for r in sa if min(r["width"], r["height"]) < 768]
    if sub: LOG.info("%s con lado menor < 768: n=%d mediana=%.3f", a.a, len(sub), np.median(an.values(sub, "C0", 0)))


if __name__ == "__main__":
    main()
