#!/usr/bin/env python3
"""L2-Mallet: correlacion de ruido por canal (Mallet 2025) sobre Synthbuster, LOGO de 8 familias, solo CPU.

Metodo (por imagen, tras crop central 512x512 SIN resize y JPEG Q90 4:2:0 en memoria):
  1. Y, Cb, Cr (float). F_c = I_c (*) L4,  L4 = [[0,1,0],[1,-4,1],[0,1,0]]   (convolucion 'valid').
  2. Dividir F_c en bloques 8x8 y vectorizarlos: b_i en R^64.
  3. B_c = {b_i : i en argmin_T Var[b_i]} U {b_i : i en argmin_T E[b_i]}     (T = 128, no lo fija el paper).
  4. R_c[u,v] = Cov(b_u, b_v) / sqrt(Var(b_u) Var(b_v)) entre las 64 posiciones del bloque (matriz 64x64 sobre B_c).
  5. SR_c = triangulo inferior de R_c (2016 valores). Feature = concat(SR_Y, SR_Cb, SR_Cr) = 6048.
Clasificador: regresion logistica L2 (class_weight balanced), C elegido por CV interna en {1e-3,1e-2,1e-1,1}.

Orden (preregistro L2, secciones 3 y 8):
  A. PUERTA DE BYTES, sin extraer features: AUC univariante de bytes (PNG del recorte y JPEG Q90) real vs fake por familia.
     Cada AUC debe estar en [0.40, 0.60]. Si alguna familia falla, el script PARA (codigo 2), salvo --allow-exclude,
     que aplica la regla del preregistro (excluir y seguir si quedan >= 5 familias).
  B. Features (cacheadas en --out/features.npz).
  C. LOGO: 8 pliegues; las reales se reparten en 8 grupos y las del grupo k no entran en el entrenamiento del pliegue k.
  D. Controles: permutaciones, baseline B0 (solo bytes), real-vs-real, y brazo sin igualar (--with-raw).
Solo lee el dataset; solo escribe en --out.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("l2-mallet")
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mallet = _load("l1_mallet", "l1-mallet.py")
corpus = _load("build_l2", "build-l2-corpus.py")
proxy = mallet.proxy

SIZE = 512
SEED = 20261111
GATE_LO, GATE_HI = 0.40, 0.60
MIN_FAMILIES = 5
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
FAMILY_OF = {"stable-diffusion-1-3": "sd1", "stable-diffusion-1-4": "sd1", "midjourney-v5": "midjourney"}
N_FAMILIES = 8


# ---------------------------------------------------------------- datos

def list_images(d):
    out = []
    for dp, _, names in os.walk(d):
        out += [os.path.join(dp, n) for n in sorted(names) if n.lower().endswith(IMG_EXT)]
    return sorted(out)


def discover(root, real_dir, max_per_family, seed):
    """Devuelve lista (path, label, family). Las carpetas de IA son los subdirectorios de root salvo la de reales."""
    real_abs = os.path.abspath(real_dir)
    items = []
    fam_files = {}
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full) or os.path.abspath(full) == real_abs or os.path.abspath(full) in real_abs:
            continue
        fam_files.setdefault(FAMILY_OF.get(d, d), []).extend(list_images(full))
    rng = np.random.default_rng(seed)
    for fam, files in sorted(fam_files.items()):
        if max_per_family and len(files) > max_per_family:
            files = [files[i] for i in sorted(rng.choice(len(files), max_per_family, replace=False))]
        items += [(f, 1, fam) for f in files]
    items += [(f, 0, "real") for f in list_images(real_dir)]
    return items


def _crop_and_encode(path):
    """Devuelve (crop RGB, info) o (None, motivo). Nunca reescala."""
    with Image.open(path) as im:
        w, h = im.size
        info = {"w": w, "h": h, "fmt": im.format, "mode": im.mode}
        if im.mode == "RGBA" and im.getchannel("A").getextrema() == (255, 255):
            # alpha totalmente opaco: conversion sin perdida (desviacion L2-D6)
            im = im.convert("RGB")
            info["alpha_opaco_descartado"] = True
        if im.mode != "RGB":
            return None, {**info, "reason": f"modo:{im.mode}"}
        if w < SIZE or h < SIZE:
            return None, {**info, "reason": "menor_que_512"}
        im.load()
        box = corpus.crop_box(w, h) if (w, h) != (SIZE, SIZE) else (0, 0, SIZE, SIZE)
        return im.crop(box).copy(), info


def jpeg_q90(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, subsampling=2, optimize=False)
    data = buf.getvalue()
    return Image.open(io.BytesIO(data)), len(data)


def bytes_pass(path):
    try:
        crop, info = _crop_and_encode(path)
        if crop is None:
            return None, info
        png = io.BytesIO()
        crop.save(png, format="PNG")
        _, jb = jpeg_q90(crop)
        return {"png_bytes": png.tell(), "jpeg_bytes": jb, **info}, None
    except Exception as exc:  # archivo corrupto
        LOG.warning("ilegible %s: %s", path, exc)
        return None, {"reason": "error_lectura"}


def feature_pass(args):
    path, T, with_raw = args
    try:
        crop, info = _crop_and_encode(path)
        if crop is None:
            return None
        eq, _ = jpeg_q90(crop)
        fe = mallet.mallet_features(eq, T)
        fr = mallet.mallet_features(crop, T) if with_raw else None
        return fe, fr
    except Exception as exc:  # archivo corrupto
        LOG.warning("ilegible %s: %s", path, exc)
        return None


def run_pool(fn, jobs, workers):
    if workers > 1:
        with Pool(workers) as p:
            return p.map(fn, jobs, chunksize=8)
    return [fn(j) for j in jobs]


# ---------------------------------------------------------------- A. puerta de bytes

def match_bytes(rows, seed, nbins=10, kmin=5):
    """Igualado de complejidad (L2-D9). Solo usa bytes PNG, nunca features.
    Contenedores = deciles de los reales. Se conservan los contenedores donde TODAS las familias
    tienen >= kmin imagenes; los reales se restringen a ellos y cada familia se submuestrea
    para reproducir la distribucion de los reales en esos contenedores."""
    rng = np.random.default_rng(seed)
    real = [r for r in rows if r["label"] == 0]
    edges = np.quantile([r["png_bytes"] for r in real], np.linspace(0, 1, nbins + 1)[1:-1])
    b = lambda r: int(np.searchsorted(edges, r["png_bytes"], side="right"))
    fams = sorted({r["fam"] for r in rows if r["label"] == 1})
    avail = {f: Counter(b(r) for r in rows if r["fam"] == f) for f in fams}
    keep = [k for k in range(nbins) if all(avail[f][k] >= kmin for f in fams)]
    info = {"edges": [float(e) for e in edges], "bins_conservados": keep,
            "disponibles": {f: [avail[f][k] for k in range(nbins)] for f in fams}}
    if len(keep) < 3:
        info["status"] = "SIN_SOPORTE_COMUN"
        return rows, info
    real_k = [r for r in real if b(r) in keep]
    share = {k: sum(1 for r in real_k if b(r) == k) / len(real_k) for k in keep}
    out = list(real_k)
    info["n_real"] = len(real_k)
    info["n_por_familia"] = {}
    for f in fams:
        fk = [r for r in rows if r["fam"] == f]
        n = min(int(avail[f][k] // share[k]) for k in keep)
        info["n_por_familia"][f] = n
        for k in keep:
            pool = [r for r in fk if b(r) == k]
            take = min(len(pool), int(round(n * share[k])))
            out += [pool[i] for i in sorted(rng.choice(len(pool), take, replace=False))]
    info["status"] = "OK"
    return out, info


def _family_stats(rows):
    real = [r for r in rows if r["label"] == 0]
    res = {}
    for fam in sorted({r["fam"] for r in rows if r["label"] == 1}):
        fk = [r for r in rows if r["fam"] == fam]
        y = [0] * len(real) + [1] * len(fk)
        a = {}
        for key in ("png_bytes", "jpeg_bytes"):
            a[key] = proxy.auc(y, [r[key] for r in real] + [r[key] for r in fk])
        a["pasa"] = all(GATE_LO <= v <= GATE_HI for v in a.values())
        a["n_fake"] = len(fk)
        res[fam] = a
    return res


def gate(items, workers, match=False, seed=SEED):
    res = run_pool(bytes_pass, [it[0] for it in items], workers)
    rows, dropped = [], Counter()
    for it, (r, why) in zip(items, res):
        if r is None:
            dropped[f"{it[2]}|{why['reason']}"] += 1
        else:
            rows.append({"path": it[0], "label": it[1], "fam": it[2], **r})
    out = {"dropped": dict(dropped), "families": {}, "formats": {}}
    if match:
        out["families_sin_igualar"] = _family_stats(rows)
        rows, out["igualado"] = match_bytes(rows, seed)
    for cls, name in ((0, "real"), (1, "fake")):
        out["formats"][name] = dict(Counter(r["fmt"] for r in rows if r["label"] == cls))
    out["families"] = _family_stats(rows)
    return rows, out


# ---------------------------------------------------------------- C. LOGO

def make_clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=300))


def pick_C(X, y, rng):
    idx = rng.permutation(len(y))[:2000]
    if len(np.unique(y[idx])) < 2:
        return 1.0
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    scores = {C: cross_val_score(make_clf(C), X[idx], y[idx], cv=skf, scoring="roc_auc").mean() for C in C_GRID}
    return max(scores, key=scores.get)


def logo(X, y, fam, grp, families, rng, permute=False, fixed_C=None):
    """Puntuaciones fuera de muestra: cada fake las da el pliegue de su familia; cada real, el de su grupo."""
    oof = np.full(len(y), np.nan)
    chosen = []
    for k, f in enumerate(families):
        test = ((fam == f) & (y == 1)) | ((y == 0) & (grp == k))
        train = ~test & ((y == 1) | (grp != k))
        ytr = y[train]
        if permute:
            ytr = rng.permutation(ytr)
        C = fixed_C if fixed_C is not None else pick_C(X[train], ytr, rng)
        chosen.append(C)
        oof[test] = make_clf(C).fit(X[train], ytr).decision_function(X[test])
    return oof, chosen


def family_aucs(y, fam, oof, families):
    real = y == 0
    out = {}
    for f in families:
        m = real | ((fam == f) & (y == 1))
        out[f] = proxy.auc(y[m], oof[m])
    return out


def boot_macro(y, fam, oof, families, rng, n=1000):
    real = np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        r = rng.choice(real, len(real))
        aucs = []
        for f in families:
            fk = np.where((fam == f) & (y == 1))[0]
            fk = rng.choice(fk, len(fk))
            idx = np.concatenate([r, fk])
            aucs.append(proxy.auc(y[idx], oof[idx]))
        vals.append(np.mean(aucs))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(y, fam, oof, families, rng, ci=True):
    per = family_aucs(y, fam, oof, families)
    macro = float(np.mean(list(per.values())))
    return {"per_family": per, "auc_macro": macro, "auc_macro_ci95": boot_macro(y, fam, oof, families, rng) if ci else None}


def real_vs_real(X, y, rng, C):
    Xr = X[y == 0]
    lab = rng.permutation(np.arange(len(Xr)) % 2)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    s = cross_val_predict(make_clf(C), Xr, lab, cv=skf, method="decision_function")
    return proxy.auc(lab, s)


# ---------------------------------------------------------------- orquestacion

def decide(res, with_raw):
    m = res["mallet_igualado"]
    lo, hi = m["auc_macro_ci95"]
    causes = []
    if m["auc_macro"] <= 0.65 or hi <= 0.70:
        causes.append("C1: AUC macro <= 0.65 o limite superior <= 0.70")
    if with_raw and res["mallet_sin_igualar"]["auc_macro"] - m["auc_macro"] > 0.15:
        causes.append("C2: la AUC cae > 0.15 al igualar (senal de compresion)")
    if res["control_real_vs_real_auc"] > 0.65:
        causes.append("C3: control real-vs-real > 0.65")
    if m["auc_macro"] - res["baseline_B0_bytes"]["auc_macro"] < 0.05:
        causes.append("C4: no supera a B0 (bytes) en >= 0.05")
    if causes:
        return "CIERRE", causes
    if m["auc_macro"] >= 0.75 and lo > 0.65 and min(m["per_family"].values()) >= 0.60:
        return "EXITO", []
    return "NO_CONCLUYENTE", []


def run(root, real_dir, out_dir, T, workers, max_per_family, n_perm, with_raw, allow_exclude, seed=SEED, expect=N_FAMILIES, match=False):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = discover(root, real_dir, max_per_family, seed)
    LOG.info("%d imagenes encontradas", len(items))
    rows, g = gate(items, workers, match, seed)
    fams_found = sorted(g["families"])
    g["familias_descubiertas"] = sorted({it[2] for it in items if it[1] == 1})
    g["familias_sin_imagenes_validas"] = sorted(set(g["familias_descubiertas"]) - set(fams_found))
    with open(os.path.join(out_dir, "gate_bytes.json"), "w") as fh:
        json.dump(g, fh, indent=2)
    if g["dropped"]:
        LOG.warning("imagenes descartadas (familia|motivo): %s", g["dropped"])
    if expect and len(fams_found) != expect:
        LOG.error("se esperaban %d familias y hay %d: %s; sin imagenes validas: %s. Ver gate_bytes.json",
                  expect, len(fams_found), fams_found, g["familias_sin_imagenes_validas"])
        return {"status": "ABORTADO_FAMILIAS", "gate": g}, 3
    if match and g["igualado"]["status"] != "OK":
        LOG.error("igualado de bytes sin soporte comun: no se extrae ninguna feature. Ver gate_bytes.json")
        return {"status": "PARADO_SIN_SOPORTE_COMUN", "gate": g}, 2
    failed = [f for f, a in g["families"].items() if not a["pasa"]]
    g["familias_fallan"] = failed
    g["formatos_distintos"] = g["formats"]["real"] != g["formats"]["fake"]
    with open(os.path.join(out_dir, "gate_bytes.json"), "w") as fh:
        json.dump(g, fh, indent=2)
    for f, a in g["families"].items():
        LOG.info("puerta bytes %-12s png=%.3f jpeg=%.3f %s", f, a["png_bytes"], a["jpeg_bytes"], "OK" if a["pasa"] else "FALLA")
    if g["formatos_distintos"]:
        LOG.warning("formatos de origen distintos entre clases: %s", g["formats"])
    families = fams_found
    if failed:
        if not allow_exclude:
            LOG.error("PUERTA DE BYTES FALLA en %s: no se extrae ninguna feature. Ver gate_bytes.json", failed)
            return {"status": "PARADO_PUERTA_BYTES", "gate": g}, 2
        families = [f for f in fams_found if f not in failed]
        if len(families) < MIN_FAMILIES:
            LOG.error("quedan %d familias (< %d): no concluyente, no se hace LOGO", len(families), MIN_FAMILIES)
            return {"status": "NO_CONCLUYENTE_POCAS_FAMILIAS", "gate": g}, 2
        LOG.warning("excluidas por la puerta (no se rescatan): %s", failed)
    rows = [r for r in rows if r["label"] == 0 or r["fam"] in families]
    y = np.array([r["label"] for r in rows])
    fam = np.array([r["fam"] for r in rows])
    paths = [r["path"] for r in rows]
    ph = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    cache = os.path.join(out_dir, "features.npz")
    FE = FR = None
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["paths_sha"]) == ph and int(z["T"]) == T and (not with_raw or "fr" in z.files):
            FE, FR = z["fe"], (z["fr"] if "fr" in z.files else None)
            LOG.info("features cargadas de cache")
    if FE is None:
        LOG.info("extrayendo features de %d imagenes (T=%d, %d procesos)", len(paths), T, workers)
        res = run_pool(feature_pass, [(p, T, with_raw) for p in paths], workers)
        ok = [i for i, r in enumerate(res) if r is not None]
        if len(ok) != len(res):
            raise RuntimeError(f"{len(res) - len(ok)} imagenes fallaron en la extraccion tras pasar la puerta")
        FE = np.stack([r[0] for r in res])
        FR = np.stack([r[1] for r in res]) if with_raw else None
        np.savez(cache, fe=FE, **({"fr": FR} if with_raw else {}), paths_sha=ph, T=T)
    grp = np.full(len(y), -1)
    real_idx = rng.permutation(np.where(y == 0)[0])
    for i, j in enumerate(real_idx):
        grp[j] = i % len(families)
    result = {"meta": {"seed": seed, "T": T, "S": SIZE, "jpeg_q": 90, "families": families, "n_real": int((y == 0).sum()),
                       "n_fake": {f: int(((fam == f) & (y == 1)).sum()) for f in families},
                       "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                       "status": "preregistro L2 (borrador); extractor Mallet con T y clasificador no fijados por el paper"},
              "gate_bytes": g}
    oof, chosen = logo(FE, y, fam, grp, families, rng)
    result["mallet_igualado"] = summarize(y, fam, oof, families, rng)
    result["mallet_igualado"]["C_elegidos"] = chosen
    Cmed = float(np.median(chosen))
    pr = result["mallet_igualado"]["per_family"]
    real_scores = oof[y == 0]
    thr = float(np.quantile(real_scores, 0.95))
    result["mallet_igualado"]["tpr_fpr5_descriptivo"] = {
        f: float((oof[(fam == f) & (y == 1)] >= thr).mean()) for f in families}
    LOG.info("AUC macro Mallet LOGO = %.4f %s | por familia %s", result["mallet_igualado"]["auc_macro"],
             result["mallet_igualado"]["auc_macro_ci95"], {k: round(v, 3) for k, v in pr.items()})
    if with_raw:
        o2, _ = logo(FR, y, fam, grp, families, rng, fixed_C=Cmed)
        result["mallet_sin_igualar"] = summarize(y, fam, o2, families, rng, ci=False)
    Xb = np.log(np.array([[r["png_bytes"], r["jpeg_bytes"]] for r in rows], float))
    ob, _ = logo(Xb, y, fam, grp, families, rng, fixed_C=1.0)
    result["baseline_B0_bytes"] = summarize(y, fam, ob, families, rng, ci=False)
    perm = [summarize(y, fam, logo(FE, y, fam, grp, families, rng, permute=True, fixed_C=Cmed)[0], families, rng, ci=False)["auc_macro"]
            for _ in range(n_perm)]
    sd = float(np.std(perm, ddof=1)) if n_perm > 1 else float("nan")
    result["control_permutaciones"] = {"n_perm": n_perm, "media": float(np.mean(perm)), "sd": sd, "se": sd / np.sqrt(n_perm) if n_perm > 1 else None}
    result["control_real_vs_real_auc"] = float(real_vs_real(FE, y, rng, Cmed))
    result["veredicto"], result["causas"] = decide(result, with_raw)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(out_dir, "predictions.csv"), "w") as fh:
        fh.write("path,family,label,score\n")
        for p, f, l, s in zip(paths, fam, y, oof):
            fh.write(f"{p},{f},{int(l)},{s:.6f}\n")
    LOG.info("VEREDICTO (preregistro seccion 8): %s %s", result["veredicto"], result["causas"])
    return result, 0


# ---------------------------------------------------------------- autotest

def _img(rng, n, corr, smooth=False):
    a = rng.normal(128, 3 if smooth else 40, (n, n, 3))
    if corr:
        from scipy.ndimage import gaussian_filter
        a = np.stack([gaussian_filter(a[..., c], 1.0) for c in range(3)], -1)
        a = (a - a.mean()) / (a.std() + 1e-9) * 40 + 128
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _synth(root, rng, n, signal, smooth_fake=False):
    names = ["dalle2", "dalle3", "firefly", "glide", "midjourney-v5", "stable-diffusion-1-3", "stable-diffusion-1-4",
             "stable-diffusion-2", "stable-diffusion-xl"]
    os.makedirs(os.path.join(root, "Raise-1k", "real_RAISE_1k"))
    for i in range(n * 4):
        _img(rng, 560, False).save(os.path.join(root, "Raise-1k", "real_RAISE_1k", f"r{i}.png"))
    for d in names:
        os.makedirs(os.path.join(root, d))
        for i in range(n):
            _img(rng, 540, signal, smooth_fake).save(os.path.join(root, d, f"f{i}.png"))


def self_test():
    ok = []

    def chk(name, cond, detail):
        ok.append(bool(cond))
        LOG.info("[%s] %s: %s", "OK" if cond else "FALLO", name, detail)

    rng = np.random.default_rng(SEED)
    chk("T1 crop sin resize", corpus.crop_box(1024, 768) == (256, 128, 768, 640), corpus.crop_box(1024, 768))
    chk("T2 agrupacion SD1.3/1.4", FAMILY_OF["stable-diffusion-1-3"] == FAMILY_OF["stable-diffusion-1-4"] == "sd1", "sd1")
    for label, signal, smooth in (("con senal", True, False), ("sin senal", False, False), ("bytes sesgados", False, True)):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "s")
            os.makedirs(root)
            _synth(root, rng, 16, signal, smooth)
            global GATE_LO, GATE_HI
            saved = (GATE_LO, GATE_HI)
            if label != "bytes sesgados":  # el ruido correlado cambia los bytes; aqui se prueba el LOGO, la puerta se prueba aparte
                GATE_LO, GATE_HI = 0.0, 1.0
            res, code = run(root, os.path.join(root, "Raise-1k", "real_RAISE_1k"), os.path.join(tmp, "o"), 128, 1, 0, 3,
                            False, False)
            GATE_LO, GATE_HI = saved
            if label == "bytes sesgados":
                chk("T5 puerta de bytes para y no extrae features", code == 2 and not os.path.exists(os.path.join(tmp, "o", "features.npz")),
                    f"codigo={code}")
            elif label == "con senal":
                a = res["mallet_igualado"]["auc_macro"]
                chk("T3 con senal: AUC macro LOGO > 0.90", code == 0 and a > 0.90, f"AUC={a:.3f} familias={len(res['meta']['families'])}")
            else:
                a = res["mallet_igualado"]["auc_macro"]
                chk("T4 sin senal: AUC macro en [0.35, 0.65]", code == 0 and 0.35 <= a <= 0.65, f"AUC={a:.3f}")
    r2 = np.random.default_rng(7)
    mk = lambda lab, fam, mu, n: [{"path": f"{fam}{i}", "label": lab, "fam": fam, "png_bytes": float(v), "jpeg_bytes": float(v) * 0.3}
                                  for i, v in enumerate(r2.normal(mu, 20, n))]
    rows = mk(0, "real", 100, 600) + mk(1, "a", 115, 600) + mk(1, "b", 90, 600)
    pre = _family_stats(rows)
    m, inf = match_bytes(rows, SEED)
    post = _family_stats(m)
    chk("T6 igualado: AUC bytes sesgado -> dentro de [0.40, 0.60]",
        inf["status"] == "OK" and not any(v["pasa"] for v in pre.values()) and all(v["pasa"] for v in post.values()),
        f"antes={ {k: round(v['png_bytes'], 2) for k, v in pre.items()} } despues={ {k: round(v['png_bytes'], 2) for k, v in post.items()} } n={inf.get('n_por_familia')}")
    _, inf2 = match_bytes(mk(0, "real", 100, 300) + mk(1, "a", 300, 300), SEED)
    chk("T7 igualado sin soporte comun se declara", inf2["status"] == "SIN_SOPORTE_COMUN", inf2["status"])
    LOG.info("autotest: %d/%d correctos", sum(ok), len(ok))
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default="/home/gustavo/dataset_test/synthbuster/synthbuster_data/synthbuster")
    ap.add_argument("--real-dir", default="/home/gustavo/dataset_test/synthbuster/synthbuster_data/synthbuster/Raise-1k/real_RAISE_1k")
    ap.add_argument("--out", default=None)
    ap.add_argument("--T", type=int, default=mallet.T_DEFAULT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-per-family", type=int, default=0, help="0 = todas")
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--with-raw", action="store_true", help="brazo sin igualar (duplica el tiempo de extraccion)")
    ap.add_argument("--allow-exclude", action="store_true", help="aplicar la regla del preregistro: excluir familias que fallan y seguir si quedan >= 5")
    ap.add_argument("--match-bytes", action="store_true", help="igualado de bytes PNG por contenedores antes de la puerta (L2-D9)")
    ap.add_argument("--expect-families", type=int, default=N_FAMILIES, help="familias con imagenes validas esperadas (0 = no comprobar)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        return self_test()
    if not args.out:
        ap.error("--out es obligatorio")
    _, code = run(args.root, args.real_dir, args.out, args.T, args.workers, args.max_per_family, args.n_perm,
                  args.with_raw, args.allow_exclude, expect=args.expect_families, match=args.match_bytes)
    return code


if __name__ == "__main__":
    sys.exit(main())
