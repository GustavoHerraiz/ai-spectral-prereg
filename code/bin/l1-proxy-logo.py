#!/usr/bin/env python3
"""L1-proxy: baseline exploratorio de huella espectral de banda baja/media con holdout por generador.

NO es una replica de Manisha 2025 (metodo no verificado): es un proxy generico
(espectro de potencia radial en [0.02, 0.35) ciclos/pixel) bajo el Pipeline de Procesado Igualado
del preregistro. No cuenta para el gate de replicacion ni para H1.

Solo lee el dataset; solo escribe en --out.
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image, JpegImagePlugin, features as pil_features
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("l1-proxy")

S = 256
K = 32
JPEG_Q = 90
F_LO, F_HI = 0.02, 0.35
SEED = 20261111
TEST_GENERATORS = ("stylegan2", "gaugan", "crn")
# Excluidos ex ante (D5): sesgo estructural de tamano/remuestreo o de compresion entre reales y falsas
EXCLUDED_GENERATORS = ("whichfaceisreal", "san", "deepfake")
FPR_TARGET = 0.05
N_BOOT = 2000
N_PERM = 100
EPS = 1e-12
IMG_EXT = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------- pipeline

def _radial_index(size, k):
    f = np.fft.fftfreq(size)
    r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    edges = np.linspace(F_LO, F_HI, k + 1)
    idx = np.digitize(r, edges) - 1
    valid = (idx >= 0) & (idx < k)
    return idx, valid


_IDX, _VALID = _radial_index(S, K)
_WIN = np.outer(np.hanning(S), np.hanning(S))


def center_crop(img, size):
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def equalize(img):
    """Pipeline igualado: RGB 8 bit -> recorte SxS -> JPEG Q90 4:2:0 (en memoria) -> decodificar."""
    rgb = center_crop(img.convert("RGB"), S)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=JPEG_Q, subsampling=2, optimize=False)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out, buf.getvalue()


def spectral_features(img):
    gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    gray = gray - gray.mean()
    power = np.abs(np.fft.fft2(gray * _WIN)) ** 2
    sums = np.bincount(_IDX[_VALID], weights=power[_VALID], minlength=K)
    counts = np.bincount(_IDX[_VALID], minlength=K)
    ring = np.log10(sums / np.maximum(counts, 1) + EPS)
    return ring - ring.mean()


def extract(path):
    """Devuelve (feat_igualado, feat_sin_igualar, meta) o (None, None, motivo)."""
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            nbytes = os.path.getsize(path)
            if min(w, h) < S:
                return None, None, "menor_que_S"
            raw = center_crop(im.convert("RGB"), S)
            feat_raw = spectral_features(raw)
            eq, _ = equalize(im)
            feat_eq = spectral_features(eq)
            return feat_eq, feat_raw, {"w": w, "h": h, "bytes": nbytes}
    except Exception as exc:  # imagen corrupta o formato no soportado
        LOG.warning("fallo leyendo %s: %s", path, exc)
        return None, None, "error_lectura"


# ---------------------------------------------------------------- dataset

def discover(root, exclude=EXCLUDED_GENERATORS):
    """Devuelve lista (generador, etiqueta, ruta). 0_real -> 0, 1_fake -> 1; recursivo."""
    items = []
    for gen in sorted(os.listdir(root)):
        gdir = os.path.join(root, gen)
        if not os.path.isdir(gdir) or gen in exclude:
            continue
        for dirpath, _, files in os.walk(gdir):
            base = os.path.basename(dirpath)
            if base not in ("0_real", "1_fake"):
                continue
            label = 0 if base == "0_real" else 1
            for fn in sorted(files):
                if fn.lower().endswith(IMG_EXT):
                    items.append((gen, label, os.path.join(dirpath, fn)))
    return items


def subsample(items, max_per_class, rng):
    if not max_per_class:
        return items
    groups = {}
    for it in items:
        groups.setdefault((it[0], it[1]), []).append(it)
    out = []
    for key in sorted(groups):
        g = groups[key]
        if len(g) > max_per_class:
            sel = rng.choice(len(g), max_per_class, replace=False)
            g = [g[i] for i in sorted(sel)]
        out.extend(g)
    return out


def build_table(items, workers):
    paths = [it[2] for it in items]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(extract, paths, chunksize=16)
    else:
        res = [extract(p) for p in paths]
    rows, excluded = [], {}
    for (gen, label, path), (fe, fr, meta) in zip(items, res):
        if fe is None:
            key = f"{gen}|{label}|{meta}"
            excluded[key] = excluded.get(key, 0) + 1
            continue
        rows.append({"gen": gen, "label": label, "path": path, "fe": fe, "fr": fr, **meta})
    return rows, excluded


# ---------------------------------------------------------------- metricas

def auc(y, s):
    y = np.asarray(y)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def boot_ci(y, s, rng, n=N_BOOT):
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a = auc(y[i], s[i])
        if not np.isnan(a):
            vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_macro_ci(per_gen, rng, n=N_BOOT):
    """IC de la AUC macro remuestreando imagenes dentro de cada generador de prueba."""
    vals = []
    for _ in range(n):
        accs = []
        for y, s in per_gen:
            i = rng.integers(0, len(y), len(y))
            a = auc(y[i], s[i])
            if not np.isnan(a):
                accs.append(a)
        vals.append(np.mean(accs))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def make_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def threshold_at_fpr(scores_neg, fpr):
    return float(np.quantile(scores_neg, 1 - fpr))


def evaluate(rows, feat_key, test_gens, rng, permute=False, size_features=False, ci=True):
    """Entrena con generadores no excluidos, prueba con los excluidos."""
    train = [r for r in rows if r["gen"] not in test_gens]
    test = [r for r in rows if r["gen"] in test_gens]
    if size_features:
        X = lambda rs: np.array([[r["w"], r["h"], r["bytes"], r["bytes"] / (r["w"] * r["h"])] for r in rs], float)
    else:
        X = lambda rs: np.array([r[feat_key] for r in rs])
    ytr = np.array([r["label"] for r in train])
    if permute:
        ytr = rng.permutation(ytr)
    idx = rng.permutation(len(train))
    n_val = max(1, int(0.2 * len(train)))
    val_i, fit_i = idx[:n_val], idx[n_val:]
    Xtr = X(train)
    clf = make_clf().fit(Xtr[fit_i], ytr[fit_i])
    val_scores = clf.decision_function(Xtr[val_i])
    val_neg = val_scores[ytr[val_i] == 0]
    thr = threshold_at_fpr(val_neg, FPR_TARGET) if len(val_neg) else 0.0
    Xte = X(test)
    yte = np.array([r["label"] for r in test])
    gte = np.array([r["gen"] for r in test])
    ste = clf.decision_function(Xte)
    per_gen, per_gen_pairs = {}, []
    for g in sorted(test_gens):
        m = gte == g
        if m.sum() == 0:
            continue
        a = auc(yte[m], ste[m])
        pred = ste[m] >= thr
        yy = yte[m]
        per_gen[g] = {
            "n_real": int((yy == 0).sum()),
            "n_fake": int((yy == 1).sum()),
            "auc": a,
            "auc_ci95": boot_ci(yy, ste[m], rng) if ci else None,
            "fpr_at_val_thr": float(pred[yy == 0].mean()) if (yy == 0).any() else None,
            "tpr_at_val_thr": float(pred[yy == 1].mean()) if (yy == 1).any() else None,
        }
        per_gen_pairs.append((yy, ste[m]))
    macro = float(np.mean([v["auc"] for v in per_gen.values()]))
    return {
        "n_train": len(train),
        "n_test": len(test),
        "val_threshold": thr,
        "per_generator": per_gen,
        "auc_macro": macro,
        "auc_macro_ci95": boot_macro_ci(per_gen_pairs, rng) if ci else None,
    }, {"gen": gte, "y": yte, "score": ste, "paths": [r["path"] for r in test]}


def run_experiment(root, out_dir, test_gens, max_per_class=0, workers=1, seed=SEED, n_perm=N_PERM):
    rng = np.random.default_rng(seed)
    items = subsample(discover(root), max_per_class, rng)
    if not items:
        raise RuntimeError(f"sin imagenes en {root}")
    gens = sorted({it[0] for it in items})
    missing = [g for g in test_gens if g not in gens]
    if missing:
        raise RuntimeError(f"generadores de prueba ausentes: {missing}")
    LOG.info("%d imagenes, %d generadores", len(items), len(gens))
    rows, excluded = build_table(items, workers)
    train_gens = [g for g in gens if g not in test_gens]
    assert not set(train_gens) & set(test_gens)

    result = {
        "meta": {
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "seed": seed, "S": S, "K": K, "jpeg_quality": JPEG_Q, "band": [F_LO, F_HI],
            "train_generators": train_gens, "test_generators": sorted(test_gens),
            "excluded_generators": list(EXCLUDED_GENERATORS),
            "pillow": Image.__version__, "libjpeg": pil_features.version("jpg"),
            "numpy": np.__version__, "max_per_class": max_per_class,
            "status": "exploratorio; proxy generico, no replica de Manisha 2025",
        },
        "counts": {g: {"real": sum(1 for r in rows if r["gen"] == g and r["label"] == 0),
                       "fake": sum(1 for r in rows if r["gen"] == g and r["label"] == 1)} for g in gens},
        "excluded": excluded,
    }
    main, preds = evaluate(rows, "fe", test_gens, rng)
    result["igualado"] = main
    result["diagnostico_sin_igualar"], _ = evaluate(rows, "fr", test_gens, rng)
    perm = [evaluate(rows, "fe", test_gens, rng, permute=True, ci=False)[0]["auc_macro"] for _ in range(n_perm)]
    result["control_etiquetas_permutadas"] = {
        "n_perm": n_perm, "auc_macro_mean": float(np.mean(perm)), "auc_macro_sd": float(np.std(perm, ddof=1)),
        "auc_macro_se": float(np.std(perm, ddof=1) / np.sqrt(n_perm)),
    }
    result["control_tamano_bytes_originales"], _ = evaluate(rows, None, test_gens, rng, size_features=True)

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

def _synth_field(rng, n, beta=1.0):
    """Campo 1/f^beta aleatorio, normalizado a [0, 255]."""
    f = np.fft.fftfreq(n)
    r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    r[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / r ** beta
    x = np.real(np.fft.ifft2(spec))
    x = (x - x.min()) / (x.max() - x.min() + EPS)
    return x * 255.0


def _make_image(rng, size, kind, pattern_freq=None, amp=0.0):
    if kind == "noise":
        a = rng.normal(128, 40, (size, size))
    else:
        a = _synth_field(rng, size)
    if pattern_freq is not None:
        yy, xx = np.mgrid[0:size, 0:size]
        a = a + amp * np.sin(2 * np.pi * pattern_freq * (xx + 0.7 * yy) + rng.uniform(0, 6.28))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")


GENS13 = ["biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle", "progan",
          "san", "seeingdark", "stargan", "stylegan", "stylegan2", "whichfaceisreal"]


def _write_dataset(root, rng, n, kind, signal, real_fmt="png"):
    for gi, g in enumerate(GENS13):
        # progan usa subcarpetas por clase, como en CNNDetection
        base = os.path.join(root, g, "airplane") if g == "progan" else os.path.join(root, g)
        for sub in ("0_real", "1_fake"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        freq = 0.18 + 0.004 * ((gi % 3) - 1)  # senal compartida (dentro de un mismo anillo) con jitter por generador
        for i in range(n):
            real = _make_image(rng, 288, kind)
            fake = _make_image(rng, 288, kind, freq if signal else None, amp=6.0)
            if real_fmt == "jpeg75":
                real.save(os.path.join(base, "0_real", f"{i}.jpg"), quality=75)
            else:
                real.save(os.path.join(base, "0_real", f"{i}.png"))
            fake.save(os.path.join(base, "1_fake", f"{i}.png"))


def self_test():
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))
        LOG.info("[%s] %s: %s", "OK" if ok else "FALLO", name, detail)

    rng = np.random.default_rng(SEED)

    # T3a: formato, tamano e idempotencia del pipeline
    img = _make_image(rng, 300, "field")
    eq, jbytes = equalize(img)
    reopened = Image.open(io.BytesIO(jbytes))
    check("T3a formato JPEG 4:2:0 y tamano SxS",
          reopened.format == "JPEG" and reopened.size == (S, S)
          and JpegImagePlugin.get_sampling(reopened) == 2,
          f"format={reopened.format} size={reopened.size} sampling={JpegImagePlugin.get_sampling(reopened)}")
    f1 = spectral_features(eq)
    png_buf = io.BytesIO()
    eq.save(png_buf, format="PNG")
    f2 = spectral_features(equalize(Image.open(io.BytesIO(png_buf.getvalue())))[0])
    d_idem = float(np.max(np.abs(f1 - f2)))
    check("T3b idempotencia: pipeline(pipeline(x)) ~ pipeline(x)", d_idem < 0.05,
          f"max|dif| de features = {d_idem:.5f} (umbral 0.05)")
    # mismo contenido guardado como PNG o como JPEG q95 -> features casi iguales tras igualar
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=95)
    f3 = spectral_features(equalize(Image.open(io.BytesIO(b.getvalue())))[0])
    f4 = spectral_features(equalize(img)[0])
    d_fmt = float(np.max(np.abs(f3 - f4)))
    check("T3c mismo contenido PNG vs JPEG q95 tras igualar", d_fmt < 0.1,
          f"max|dif| de features = {d_fmt:.5f} (umbral 0.1)")

    with tempfile.TemporaryDirectory() as tmp:
        # T1: con senal debe separar generadores no vistos
        root1 = os.path.join(tmp, "signal")
        _write_dataset(root1, rng, 30, "field", True)
        r1 = run_experiment(root1, os.path.join(tmp, "out1"), TEST_GENERATORS, seed=1, n_perm=30)
        a1 = r1["igualado"]["auc_macro"]
        check("T1 con senal: AUC macro en generadores no vistos > 0.90", a1 > 0.90, f"AUC macro = {a1:.4f}")

        # T4: sin solapamiento
        tr, te = set(r1["meta"]["train_generators"]), set(r1["meta"]["test_generators"])
        check("T4 sin solapamiento; 7 train / 3 test; excluidos ausentes",
              not tr & te and len(tr) == 7 and len(te) == 3 and not (tr | te) & set(EXCLUDED_GENERATORS)
              and not set(r1["counts"]) & set(EXCLUDED_GENERATORS),
              f"train={len(tr)} test={len(te)} interseccion={sorted(tr & te)} excluidos_en_counts={sorted(set(r1['counts']) & set(EXCLUDED_GENERATORS))}")

        # T5: salidas
        out1 = os.path.join(tmp, "out1")
        with open(os.path.join(out1, "predictions.csv")) as fh:
            header = next(csv.reader(fh))
        mj = json.load(open(os.path.join(out1, "metrics.json")))
        need = {"igualado", "diagnostico_sin_igualar", "control_etiquetas_permutadas",
                "control_tamano_bytes_originales", "counts", "excluded", "meta"}
        check("T5 salidas completas", header == ["path", "generator", "label", "score"] and need <= set(mj),
              f"cabecera={header} claves_faltantes={sorted(need - set(mj))}")
        check("T5b progan con subcarpeta descubierto",
              r1["counts"]["progan"]["real"] == 30 and r1["counts"]["progan"]["fake"] == 30,
              f"progan={r1['counts']['progan']}")

        # T2: ruido blanco en ambas clases -> ~0.5; tambien el control de permutacion
        root2 = os.path.join(tmp, "noise")
        _write_dataset(root2, rng, 40, "noise", False)
        r2 = run_experiment(root2, os.path.join(tmp, "out2"), TEST_GENERATORS, seed=2)
        a2 = r2["igualado"]["auc_macro"]
        check("T2 ruido blanco sin senal: AUC macro en [0.40, 0.60]", 0.40 <= a2 <= 0.60, f"AUC macro = {a2:.4f}")
        pc = r1["control_etiquetas_permutadas"]
        dev = abs(pc["auc_macro_mean"] - 0.5)
        check("T2b etiquetas permutadas (30 perm., dataset con senal): media a <= 2 SE de 0.5", dev <= 2 * pc["auc_macro_se"],
              f"media = {pc['auc_macro_mean']:.4f}, sd = {pc['auc_macro_sd']:.4f}, SE = {pc['auc_macro_se']:.4f}")

        # T6 (informativo): sesgo de formato; reales JPEG q75, falsas PNG, sin senal real
        root3 = os.path.join(tmp, "fmt")
        _write_dataset(root3, rng, 40, "field", False, real_fmt="jpeg75")
        r3 = run_experiment(root3, os.path.join(tmp, "out3"), TEST_GENERATORS, seed=3, n_perm=2)
        raw, eqd = r3["diagnostico_sin_igualar"]["auc_macro"], r3["igualado"]["auc_macro"]
        LOG.info("HALLAZGO T6 (informativo, no puntua): reales JPEG q75 vs falsas PNG sin senal real; "
                 "AUC espuria sin igualar = %.4f, con igualado Q%d = %.4f", raw, JPEG_Q, eqd)

    failed = [c for c in checks if not c[1]]
    LOG.info("autotest: %d/%d correctos", len(checks) - len(failed), len(checks))
    return 0 if not failed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default="/home/gustavo/dataset_test/cnn_synth_testset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-per-class", type=int, default=0, help="submuestreo por generador/clase (0 = todo)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        return self_test()
    if not args.out:
        ap.error("--out es obligatorio fuera de --self-test")
    res = run_experiment(args.root, args.out, TEST_GENERATORS, args.max_per_class, args.workers)
    LOG.info("AUC macro igualado = %.4f (IC95 %s)", res["igualado"]["auc_macro"], res["igualado"]["auc_macro_ci95"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
