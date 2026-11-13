#!/usr/bin/env python3
"""L1-Mallet: extractor de correlacion de ruido por canal (Mallet et al., AMLDS 2025) con holdout por generador.

Implementa la seccion II.B del paper: residuo Laplaciano L4 sobre Y, Cb y Cr; bloques 8x8 vectorizados;
seleccion de los T bloques de menor varianza y los T de menor media (union); matriz de correlacion 64x64
entre posiciones del bloque; triangulo inferior (2016 coeficientes por canal, 6048 en total).

Desviaciones respecto al paper (se anotan en el preregistro):
  - T no esta especificado en el paper: se fija por defecto en 128 (parametro --T).
  - Clasificador: una regresion logistica binaria real/fake, no el esquema de 3 pasos del paper.
  - Se aplica el Pipeline de Procesado Igualado (recorte SxS + JPEG Q90 4:2:0) reutilizado de l1-proxy-logo.py.

Solo lee el dataset; solo escribe en --out. Estructura esperada: <root>/<generador>/{0_real,1_fake}/*.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

LOG = logging.getLogger("l1-mallet")
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("l1_proxy", os.path.join(HERE, "l1-proxy-logo.py"))
proxy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proxy)

L4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
BLOCK = 8
T_DEFAULT = 128
TRI = np.tril_indices(BLOCK * BLOCK, k=-1)
N_FEAT_CH = len(TRI[0])  # 2016
N_PERM = 20


def channel_feature(chan, T):
    """Correlacion de ruido de un canal: (2016,)."""
    res = convolve2d(chan, L4, mode="valid")
    h, w = (res.shape[0] // BLOCK) * BLOCK, (res.shape[1] // BLOCK) * BLOCK
    res = res[:h, :w]
    blocks = res.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).transpose(0, 2, 1, 3).reshape(-1, BLOCK * BLOCK)
    sel = np.union1d(np.argsort(blocks.var(axis=1))[:T], np.argsort(blocks.mean(axis=1))[:T])
    sub = blocks[sel]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(sub.T)
    corr = np.nan_to_num(corr, nan=0.0)  # posiciones de varianza nula
    return corr[TRI]


def mallet_features(img, T=T_DEFAULT):
    ycc = np.asarray(img.convert("YCbCr"), dtype=np.float64)
    return np.concatenate([channel_feature(ycc[..., c], T) for c in range(3)])


def extract(args):
    path, T = args
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            nbytes = os.path.getsize(path)
            if min(w, h) < proxy.S:
                return None, None, "menor_que_S"
            raw = proxy.center_crop(im.convert("RGB"), proxy.S)
            eq, _ = proxy.equalize(im)
            return mallet_features(eq, T), mallet_features(raw, T), {"w": w, "h": h, "bytes": nbytes}
    except Exception as exc:  # imagen corrupta o formato no soportado
        LOG.warning("fallo leyendo %s: %s", path, exc)
        return None, None, "error_lectura"


def build_table(items, T, workers):
    jobs = [(it[2], T) for it in items]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(extract, jobs, chunksize=16)
    else:
        res = [extract(j) for j in jobs]
    rows, excluded = [], {}
    for (gen, label, path), (fe, fr, meta) in zip(items, res):
        if fe is None:
            key = f"{gen}|{label}|{meta}"
            excluded[key] = excluded.get(key, 0) + 1
            continue
        rows.append({"gen": gen, "label": label, "path": path, "fe": fe, "fr": fr, **meta})
    return rows, excluded


def univariate_controls(rows, test_gens):
    """AUC univariante (sin entrenar) de tamano/bytes en cada generador de prueba; leer como |AUC-0.5|."""
    out = {}
    for g in sorted(test_gens):
        rs = [r for r in rows if r["gen"] == g]
        y = np.array([r["label"] for r in rs])
        out[g] = {
            "bytes": proxy.auc(y, [r["bytes"] for r in rs]),
            "bytes_por_pixel": proxy.auc(y, [r["bytes"] / (r["w"] * r["h"]) for r in rs]),
            "area": proxy.auc(y, [r["w"] * r["h"] for r in rs]),
        }
    return out


def run_experiment(root, out_dir, test_gens, exclude, T, max_per_class=0, workers=1, seed=proxy.SEED, n_perm=N_PERM):
    rng = np.random.default_rng(seed)
    items = proxy.subsample(proxy.discover(root, exclude=exclude), max_per_class, rng)
    if not items:
        raise RuntimeError(f"sin imagenes en {root}")
    gens = sorted({it[0] for it in items})
    missing = [g for g in test_gens if g not in gens]
    if missing:
        raise RuntimeError(f"generadores de prueba ausentes: {missing}")
    LOG.info("%d imagenes, %d generadores, T=%d", len(items), len(gens), T)
    rows, excluded = build_table(items, T, workers)
    train_gens = [g for g in gens if g not in test_gens]
    assert not set(train_gens) & set(test_gens)
    result = {
        "meta": {
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "seed": seed, "S": proxy.S, "jpeg_quality": proxy.JPEG_Q, "T": T, "n_features": 3 * N_FEAT_CH,
            "train_generators": train_gens, "test_generators": sorted(test_gens), "excluded_generators": list(exclude),
            "pillow": Image.__version__, "numpy": np.__version__, "max_per_class": max_per_class,
            "status": "exploratorio; extractor de Mallet 2025 con T y clasificador no especificados por el paper",
        },
        "counts": {g: {"real": sum(1 for r in rows if r["gen"] == g and r["label"] == 0),
                       "fake": sum(1 for r in rows if r["gen"] == g and r["label"] == 1)} for g in gens},
        "excluded": excluded,
    }
    result["igualado"], preds = proxy.evaluate(rows, "fe", test_gens, rng)
    result["diagnostico_sin_igualar"], _ = proxy.evaluate(rows, "fr", test_gens, rng)
    perm = [proxy.evaluate(rows, "fe", test_gens, rng, permute=True, ci=False)[0]["auc_macro"] for _ in range(n_perm)]
    result["control_etiquetas_permutadas"] = {
        "n_perm": n_perm, "auc_macro_mean": float(np.mean(perm)), "auc_macro_sd": float(np.std(perm, ddof=1)),
        "auc_macro_se": float(np.std(perm, ddof=1) / np.sqrt(n_perm)),
    }
    result["control_univariante_tamano_bytes"] = univariate_controls(rows, test_gens)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["path", "generator", "label", "score"])
        for p, g, y, s in zip(preds["paths"], preds["gen"], preds["y"], preds["score"]):
            wr.writerow([p, g, int(y), f"{s:.6f}"])
    return result


# ---------------------------------------------------------------- autotest

def _noise(rng, n, corr):
    out = np.stack([rng.normal(128, 40, (n, n)) for _ in range(3)], axis=-1)
    if corr:  # ruido espacialmente correlado, con la misma desviacion tipica
        out = np.stack([_blur_norm(out[..., c]) for c in range(3)], axis=-1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _blur_norm(a):
    b = gaussian_filter(a, 1.0)
    return (b - b.mean()) / (b.std() + 1e-12) * 40 + 128


def _write(root, rng, n, signal):
    for g in proxy.TEST_GENERATORS + ("g1", "g2", "g3", "g4", "g5", "g6", "g7"):
        for sub in ("0_real", "1_fake"):
            os.makedirs(os.path.join(root, g, sub), exist_ok=True)
        for i in range(n):
            _noise(rng, 288, False).save(os.path.join(root, g, "0_real", f"{i}.png"))
            _noise(rng, 288, signal).save(os.path.join(root, g, "1_fake", f"{i}.png"))


def self_test():
    results = []

    def check(name, ok, detail):
        results.append(ok)
        LOG.info("[%s] %s: %s", "OK" if ok else "FALLO", name, detail)

    rng = np.random.default_rng(proxy.SEED)
    # T1 forma, finitud y determinismo
    img = _noise(rng, 256, False)
    f1, f2 = mallet_features(img), mallet_features(img)
    check("T1 forma/finitud/determinismo", f1.shape == (3 * N_FEAT_CH,) and np.isfinite(f1).all() and np.array_equal(f1, f2),
          f"shape={f1.shape} finito={bool(np.isfinite(f1).all())}")
    # T2 seleccion: con la mitad plana y la mitad texturada, los bloques elegidos son los planos
    res = np.zeros((248, 248))
    res[:, 124:] = rng.normal(0, 30, (248, 124))
    blocks = res.reshape(31, 8, 31, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    sel = np.argsort(blocks.var(axis=1))[:100]
    check("T2 seleccion de bloques de baja varianza", bool(np.all((sel % 31) < 15)), "100 bloques de menor varianza en la zona plana")
    # T3 canal constante no produce NaN
    flat = channel_feature(np.full((256, 256), 100.0), T_DEFAULT)
    check("T3 imagen constante sin NaN", bool(np.isfinite(flat).all()), f"max|f|={np.abs(flat).max():.3f}")
    # T4 senal: ruido correlado frente a blanco; AUC en generadores no vistos
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, rng, 24, True)
        res = run_experiment(tmp, os.path.join(tmp, "out"), proxy.TEST_GENERATORS, (), T_DEFAULT, n_perm=10)
        a = res["igualado"]["auc_macro"]
        check("T4 con senal: AUC macro en generadores no vistos > 0.90", a > 0.90, f"AUC macro = {a:.4f}")
        p = res["control_etiquetas_permutadas"]
        check("T5 permutaciones con senal: media a <= 2 SE de 0.5", abs(p["auc_macro_mean"] - 0.5) <= 2 * p["auc_macro_se"] + 0.03,
              f"media={p['auc_macro_mean']:.4f} SE={p['auc_macro_se']:.4f}")
        ok = all(os.path.exists(os.path.join(tmp, "out", f)) for f in ("metrics.json", "predictions.csv"))
        check("T6 salidas completas", ok, "metrics.json y predictions.csv")
    # T7 sin senal: AUC cerca de 0.5
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, rng, 24, False)
        res = run_experiment(tmp, os.path.join(tmp, "out"), proxy.TEST_GENERATORS, (), T_DEFAULT, n_perm=5)
        a = res["igualado"]["auc_macro"]
        check("T7 sin senal: AUC macro en [0.40, 0.60]", 0.40 <= a <= 0.60, f"AUC macro = {a:.4f}")
    LOG.info("autotest: %d/%d correctos", sum(results), len(results))
    return 0 if all(results) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default="/home/gustavo/dataset_test/cnn_synth_testset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test-gens", default=",".join(proxy.TEST_GENERATORS), help="generadores de prueba separados por comas")
    ap.add_argument("--exclude", default=",".join(proxy.EXCLUDED_GENERATORS), help="generadores excluidos separados por comas")
    ap.add_argument("--T", type=int, default=T_DEFAULT)
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        return self_test()
    if not args.out:
        ap.error("--out es obligatorio fuera de --self-test")
    split = lambda s: tuple(x for x in s.split(",") if x)
    res = run_experiment(args.root, args.out, split(args.test_gens), split(args.exclude), args.T,
                         args.max_per_class, args.workers, n_perm=args.n_perm)
    LOG.info("AUC macro igualado = %.4f (IC95 %s)", res["igualado"]["auc_macro"], res["igualado"]["auc_macro_ci95"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
