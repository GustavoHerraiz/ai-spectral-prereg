#!/usr/bin/env python3
"""H2 etapa 2: ¿el puntaje de sd-turbo sin cadena (A1) sigue a un descriptor simple de nitidez/alta frecuencia? Ver PREREGISTRO_H2.md seccion 10.

Para cada una de las 90 imagenes de H2 etapa 1 (30 semillas x 1, 2 y 4 pasos): imagen A1 (recorte alineado + JPEG q92, igual que h1-resample),
puntaje = decision_function del modelo Synthbuster A5 vs todos los R1 (res_spec), y dos descriptores de A1: d_lap (log10 de la varianza del laplaciano
del gris) y d_hf (fraccion de energia espectral con radio > 0,25 ciclos/px, ventana de Hann). Estadistico: Spearman agrupado puntaje~descriptor,
IC95 por bootstrap de semillas (las tres condiciones de una semilla se remuestrean juntas).
Uso:
  python3 h2-descriptors.py --s1-dir D1 --s2-dir D2 --s4-dir D4 --resample-feats F --out OUT
  python3 h2-descriptors.py --self-test
"""
import argparse, json, logging, os, sys
import numpy as np
from scipy.stats import spearmanr
from PIL import Image
import importlib.util

LOG = logging.getLogger("h2-descriptors")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


STEPS = load_module("h2steps", os.path.join(HERE, "h2-steps.py"))
SD, CH = STEPS.SD, STEPS.CH
RS_PATH = os.path.join(HERE, "h1-resample.py")
RHO_EXPLICA, CI_EXPLICA, RHO_NO = 0.6, 0.4, 0.3
HF_RADIO = 0.25
DESC = ("d_lap", "d_hf")


def gray(img):
    return np.asarray(img.convert("L"), dtype=np.float64)


def d_lap(img):
    g = gray(img)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.log10(lap.var() + 1e-12))


def d_hf(img):
    g = gray(img); g = g - g.mean()
    h, w = g.shape
    p = np.abs(np.fft.fft2(g * np.outer(np.hanning(h), np.hanning(w)))) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    r = np.hypot(fy, fx)
    return float(p[r > HF_RADIO].sum() / p.sum())


def corr_stats(score, desc, seed_idx, n_boot, seed=20261111):
    """Spearman agrupado e IC95 remuestreando semillas (todas las condiciones de una semilla juntas)."""
    score, desc, seed_idx = map(np.asarray, (score, desc, seed_idx))
    rho = float(spearmanr(score, desc)[0])
    rng = np.random.default_rng(seed); us = np.unique(seed_idx); ix = {s: np.where(seed_idx == s)[0] for s in us}; v = []
    for _ in range(n_boot):
        sel = np.concatenate([ix[us[i]] for i in rng.choice(len(us), len(us))])
        v.append(spearmanr(score[sel], desc[sel])[0])
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return rho, [float(lo), float(hi)]


def verdict(stats):
    """stats: {descriptor: (rho, [lo, hi])}. 'explica' si algun descriptor tiene |rho| >= 0,6 y el IC95 queda a >= 0,4 de cero."""
    for d, (rho, ci) in stats.items():
        edge = ci[0] if rho > 0 else -ci[1]
        if abs(rho) >= RHO_EXPLICA and edge >= CI_EXPLICA:
            return f"el descriptor {d} sigue al puntaje"
    if all(abs(r) < RHO_NO for r, _ in stats.values()):
        return "ningun descriptor sigue al puntaje"
    return "indeterminada"


def analyze(a):
    rs = CH.read(a.resample_feats)
    syn = [r for r in rs if r["corpus"] == "ai"]; reals = [r for r in rs if r["corpus"] != "ai"]
    X_syn = np.array([r["arms"]["A5"][SD.FEAT] for r in syn]); X_r1 = np.array([r["arms"]["R1"][SD.FEAT] for r in reals])
    model = CH.fit(X_syn, X_r1)
    RS = load_module("h1resample", RS_PATH)
    rows = []
    for cond, d in (("1 paso", a.s1_dir), ("2 pasos", a.s2_dir), ("4 pasos", a.s4_dir)):
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".png"):
                continue
            with Image.open(os.path.join(d, fn)) as im:
                if im.size != (512, 512):
                    raise ValueError(f"{fn}: {im.size}")
                c = RS.CTL.aligned_crop(im.convert("RGB"))
            a1 = RS.jpeg(c, 92)
            sc = float(model.decision_function(np.array([RS.feats(a1)["res_spec"]]))[0])
            rows.append({"cond": cond, "seed_idx": int(fn.split("_")[1].split(".")[0]), "score": sc, "d_lap": d_lap(a1), "d_hf": d_hf(a1)})
    LOG.info("%d imagenes", len(rows))
    out = {"n": len(rows), "por_condicion": {}, "correlaciones": {}}
    LOG.info("| condicion | puntaje med | d_lap med | d_hf med |")
    for cond in ("1 paso", "2 pasos", "4 pasos"):
        m = [r for r in rows if r["cond"] == cond]
        out["por_condicion"][cond] = {k: float(np.median([r[k] for r in m])) for k in ("score",) + DESC}
        LOG.info("| %-8s | %.3f | %.3f | %.4f |", cond, *(out["por_condicion"][cond][k] for k in ("score",) + DESC))
    stats = {}
    for d in DESC:
        stats[d] = corr_stats([r["score"] for r in rows], [r[d] for r in rows], [r["seed_idx"] for r in rows], a.n_boot)
        out["correlaciones"][d] = {"rho": stats[d][0], "ic95": stats[d][1]}
        LOG.info("Spearman puntaje~%s = %.3f IC95 [%.3f, %.3f]", d, stats[d][0], *stats[d][1])
    out["veredicto"] = verdict(stats)
    LOG.info("VEREDICTO H2-E2: %s", out["veredicto"])
    with open(os.path.join(a.out, "h2-descriptors.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h2-descriptors.json"))
    return out


def self_test():
    from PIL import ImageFilter
    rng = np.random.default_rng(3)
    noise = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    blur = noise.filter(ImageFilter.GaussianBlur(2))
    assert d_hf(blur) < d_hf(noise) and d_lap(blur) < d_lap(noise), "descriptores vs desenfoque"
    n = 90; seeds = np.tile(np.arange(30), 3); desc = rng.normal(0, 1, n)
    r = {"d_lap": corr_stats(desc * 2 + rng.normal(0, 0.5, n), desc, seeds, 200), "d_hf": corr_stats(rng.normal(0, 1, n), rng.normal(0, 1, n), seeds, 200)}
    assert verdict(r) == "el descriptor d_lap sigue al puntaje", r
    r = {"d_lap": corr_stats(rng.normal(0, 1, n), desc, seeds, 200), "d_hf": corr_stats(rng.normal(0, 1, n), rng.normal(0, 1, n), seeds, 200)}
    assert verdict(r) == "ningun descriptor sigue al puntaje", r
    r = {"d_lap": corr_stats(desc + rng.normal(0, 2, n), desc, seeds, 200)}
    assert verdict(r) == "indeterminada", r
    LOG.info("autotest OK (4 comprobaciones)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("s1-dir", "s2-dir", "s4-dir", "resample-feats", "out"):
        p.add_argument("--" + k)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("s1_dir", "s2_dir", "s4_dir", "resample_feats", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    os.makedirs(a.out, exist_ok=True)
    analyze(a)


if __name__ == "__main__":
    main()
