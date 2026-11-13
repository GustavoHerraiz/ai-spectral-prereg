"""Bateria de controles post hoc sobre SD2 (sonda v1, brazo C0, jpegPhaseRange). Exploratorio; predicciones en PREREGISTRO seccion 11c.
A) B/px igualado por deciles; B) regresion OLS con familia + log(B/px) + MP + aspecto (IC bootstrap); C) pares con el mismo nombre de archivo (Wilcoxon).
Uso: python3 v1-pipeline-probe-sd2-controls.py --run-dir DIR [--a stable-diffusion-2 --b stable-diffusion-xl]"""
import argparse, importlib.util, logging, os, sys
import numpy as np
from scipy.stats import wilcoxon

LOG = logging.getLogger("sd2-ctrl")
spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1-pipeline-probe-analyze.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
SEED, N_BOOT = 20261111, 2000


def covariates(recs):
    bpp = np.array([r["bytes"] / (r["width"] * r["height"]) for r in recs])
    mp = np.array([r["width"] * r["height"] / 1e6 for r in recs])
    asp = np.array([max(r["width"], r["height"]) / min(r["width"], r["height"]) for r in recs])
    return bpp, mp, asp


def matched_bpp(xa, ba, xb, bb, rng):
    """Deciles de B/px de A; en cada bin se submuestrea al menor n entre A y B (sin mirar la feature)."""
    edges = np.quantile(ba, np.linspace(0, 1, 6))
    edges[0], edges[-1] = -np.inf, np.inf
    ia, ib = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = np.where((ba > lo) & (ba <= hi))[0]; b = np.where((bb > lo) & (bb <= hi))[0]
        n = min(len(a), len(b))
        if n >= 5:
            ia += list(rng.choice(a, n, replace=False)); ib += list(rng.choice(b, n, replace=False))
    return ia, ib


def ols_effect(y, fam, bpp, mp, asp):
    X = np.column_stack([np.ones(len(y)), fam, np.log(bpp), mp, asp])
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def run(fa, fb, name_a, name_b):
    rng = np.random.default_rng(SEED)
    xa, xb = an.values(fa, "C0", 0), an.values(fb, "C0", 0)
    ba, mpa, aa = covariates(fa); bb, mpb, ab = covariates(fb)
    out = {"crudo_mediana_a": float(np.median(xa)), "crudo_mediana_b": float(np.median(xb)), "auc_a_vs_b_crudo": an.auc(xa, xb)}
    LOG.info("crudo: mediana %s=%.3f %s=%.3f; AUC(%s>%s)=%.3f", name_a, np.median(xa), name_b, np.median(xb), name_a, name_b, out["auc_a_vs_b_crudo"])
    ia, ib = matched_bpp(xa, ba, xb, bb, rng)
    if len(ia) >= 20:
        ma, mb = xa[ia], xb[ib]
        bs = [an.auc(rng.choice(ma, len(ma)), rng.choice(mb, len(mb))) for _ in range(N_BOOT)]
        out["A_auc"] = an.auc(ma, mb); out["A_n"] = len(ia)
        LOG.info("A) B/px igualado n=%d por familia: mediana %.3f vs %.3f; AUC(%s>%s)=%.3f [%.3f, %.3f]", len(ia), np.median(ma), np.median(mb), name_a, name_b, out["A_auc"], *np.percentile(bs, [2.5, 97.5]))
    else:
        LOG.warning("A) pocos pares tras igualar (%d)", len(ia))
    y = np.concatenate([xa, xb]); fam = np.concatenate([np.ones(len(xa)), np.zeros(len(xb))])
    bpp, mp, asp = np.concatenate([ba, bb]), np.concatenate([mpa, mpb]), np.concatenate([aa, ab])
    eff = ols_effect(y, fam, bpp, mp, asp)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y)); bs.append(ols_effect(y[i], fam[i], bpp[i], mp[i], asp[i]))
    out["B_efecto"] = float(eff)
    LOG.info("B) regresion: efecto %s (vs %s) = %+.3f [%.3f, %.3f] tras ajustar log(B/px), MP, aspecto", name_a, name_b, eff, *np.percentile(bs, [2.5, 97.5]))
    stem = lambda r: os.path.splitext(os.path.basename(r["rel"]))[0]
    da, db = {stem(r): v for r, v in zip(fa, xa)}, {stem(r): v for r, v in zip(fb, xb)}
    shared = sorted(set(da) & set(db))
    out["C_pares"] = len(shared)
    if len(shared) >= 10:
        d = np.array([da[k] - db[k] for k in shared])
        p = wilcoxon(d).pvalue
        out["C_mediana_dif"] = float(np.median(d)); out["C_frac_a_menor"] = float(np.mean(d < 0))
        LOG.info("C) %d pares con mismo nombre: mediana(%s - %s)=%+.3f; %s menor en %.0f%%; Wilcoxon p=%.3g", len(shared), name_a, name_b, np.median(d), name_a, 100 * np.mean(d < 0), p)
    else:
        LOG.warning("C) solo %d pares con mismo nombre: el control de contenido no es evaluable", len(shared))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", required=True)
    p.add_argument("--a", default="stable-diffusion-2"); p.add_argument("--b", default="stable-diffusion-xl")
    a = p.parse_args()
    records, _ = an.load(a.run_dir)
    _, fams = an.split(records)
    if a.a not in fams or a.b not in fams:
        LOG.error("familia ausente"); sys.exit(1)
    run(fams[a.a], fams[a.b], a.a, a.b)


if __name__ == "__main__":
    main()
