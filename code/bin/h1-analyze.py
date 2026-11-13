#!/usr/bin/env python3
"""Analisis de H1-PRNU y H1-L2 sobre la salida de h1-extract.py. Plan en PREREGISTRO_H1.md seccion 3 (fijado antes de ver datos).
Pasos:
  --mapping-only : imprime la tabla dispositivo -> K (sin leer ninguna puntuacion) para revisarla.
  (sin flag)     : P0, P1 (PRNU) y L2 (LOFO x dispositivos), controles y veredictos. Escribe h1-analysis.json en --run-dir.
  --self-test
Uso: python3 h1-analyze.py --run-dir DIR [--mapping-only] [--n-boot 2000] [--n-perm 20]
"""
import argparse, collections, json, logging, os, re, sys, tempfile
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("h1-analyze")
SEED = 20261111
THRESH = 0.015          # umbral de v3, congelado
P0_MIN_AUC = 0.80
SUPPORT_AUC, SUPPORT_LO, FORMAT_CTRL_MIN, DEVID_MAX_ACC = 0.70, 0.60, 0.60, 0.10
N_DEV_FOLDS = 5
TOKEN = re.compile(r"(?<![A-Za-z0-9])([CD])(\d{2})(?![0-9])")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(run_dir):
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    recs = {}
    with open(os.path.join(run_dir, "h1-features.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); recs[r["key"]] = r
    ok = [r for r in recs.values() if r["status"] == "ok"]
    LOG.info("registros: %d ok, %d con error", len(ok), len(recs) - len(ok))
    return manifest, ok


def device_to_entry(device, k_ids):
    """Dresden: igualdad exacta de entryId. VISION/Modern: token C## o D## (sin prefijo dresden). 0 -> ausente, 1 -> mapeada, >1 -> ambigua."""
    if device.startswith("dresden_"):
        hits = [k for k in k_ids if k == device]
    else:
        m = TOKEN.search(device)
        hits = [k for k in k_ids if not k.startswith("dresden_") and m and any((t.group(1), t.group(2)) == (m.group(1), m.group(2)) for t in TOKEN.finditer(k))]
    return hits


def build_mapping(recs, k_ids):
    devices = sorted({r["device"] for r in recs if r["label"] == "real" and r.get("device")})
    table = {d: device_to_entry(d, k_ids) for d in devices}
    status = {d: ("ausente" if not h else "mapeada" if len(h) == 1 else "ambigua") for d, h in table.items()}
    return table, status


def print_mapping(table, status, recs):
    n = collections.Counter(r["device"] for r in recs if r["label"] == "real")
    for d in sorted(table):
        LOG.info("MAPEO %-40s n=%-3d %-8s %s", d, n[d], status[d], table[d] if table[d] else "-")
    LOG.info("MAPEO resumen: %s", dict(collections.Counter(status.values())))


# ------------------------------------------------------------------ PRNU

def boot_auc(pos, neg, rng, n_boot):
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def analyze_prnu(recs, manifest, table, status, rng, n_boot):
    ids = manifest["k_ids"]; col = {k: i for i, k in enumerate(ids)}
    out = {}
    reals = [r for r in recs if r["label"] == "real" and r["group"] == "real" and r["prnu"]]
    ai = [r for r in recs if r["label"] == "fake" and r["prnu"]]
    dres = [r for r in recs if r["group"] == "real-dresden" and r["prnu"]]
    mapped = [r for r in reals if status.get(r["device"]) == "mapeada"]
    own, other = [], []
    for r in mapped:
        s = np.array(r["prnu"]); j = col[table[r["device"]][0]]
        own.append(s[j]); other += list(np.delete(s, j))
    top1 = float(np.mean([int(np.argmax(r["prnu"]) == col[table[r["device"]][0]]) for r in mapped])) if mapped else None
    p0 = auc(own, other) if own else None
    out["P0"] = {"n_reales_con_K": len(mapped), "auc_propia_vs_ajenas": p0, "top1": top1, "valido": bool(p0 is not None and p0 >= P0_MIN_AUC)}
    LOG.info("P0 (control positivo): n=%d AUC propia>ajenas=%s top1=%s -> %s", len(mapped), None if p0 is None else round(p0, 4), top1, "VALIDO" if out["P0"]["valido"] else "INVALIDO: se detiene")
    if not out["P0"]["valido"]:
        return out
    smax = lambda rs: np.array([max(r["prnu"]) for r in rs])
    s_ai = smax(ai)
    groups = {"real_con_K": mapped, "real_camara_ausente": [r for r in reals if status.get(r["device"]) == "ausente"]}
    out["P1"] = {"ai_frac_sobre_umbral": float(np.mean(s_ai >= THRESH)), "ai_mediana_max": float(np.median(s_ai)), "n_ai": len(ai)}
    LOG.info("P1 IA: max NCC mediana=%.4f; fraccion >= %.3f: %.3f", np.median(s_ai), THRESH, np.mean(s_ai >= THRESH))
    for name, rs in groups.items():
        if len(rs) < 5:
            LOG.warning("P1 %s: n=%d insuficiente", name, len(rs)); out["P1"][name] = {"n": len(rs)}; continue
        s = smax(rs); a = auc(s, s_ai)
        out["P1"][name] = {"n": len(rs), "auc_real_alto_vs_ia": a, "ic95": boot_auc(s, s_ai, rng, n_boot), "frac_sobre_umbral": float(np.mean(s >= THRESH)), "mediana_max": float(np.median(s))}
        LOG.info("P1 %-20s n=%d AUC(real>IA)=%.3f [%.3f, %.3f] fraccion>=umbral=%.3f mediana max=%.4f", name, len(rs), a, *out["P1"][name]["ic95"], np.mean(s >= THRESH), np.median(s))
    fam = collections.defaultdict(list)
    for r in ai:
        fam[r["group"]].append(max(r["prnu"]))
    out["P1"]["ia_por_familia_frac_umbral"] = {f: float(np.mean(np.array(v) >= THRESH)) for f, v in sorted(fam.items())}
    LOG.info("P1 IA por familia, fraccion >= umbral: %s", {f: round(v, 3) for f, v in out["P1"]["ia_por_familia_frac_umbral"].items()})
    if dres:  # sensibilidad: posible solape con las fotos de la K
        dm = [r for r in dres if r["device"] in col or device_to_entry(r["device"], ids)]
        if dm:
            sd = smax(dm); out["P1"]["sensibilidad_dresden"] = {"n": len(dm), "auc_real_alto_vs_ia": auc(sd, s_ai), "frac_sobre_umbral": float(np.mean(sd >= THRESH))}
            LOG.info("P1 sensibilidad Dresden (posible solape con K): n=%d AUC=%.3f fraccion>=umbral=%.3f", len(dm), out["P1"]["sensibilidad_dresden"]["auc_real_alto_vs_ia"], out["P1"]["sensibilidad_dresden"]["frac_sobre_umbral"])
    return out


# ------------------------------------------------------------------ L2

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))


def device_folds(devices, rng):
    uniq = sorted(set(devices)); perm = rng.permutation(len(uniq))
    fold_of = {uniq[i]: k % N_DEV_FOLDS for k, i in enumerate(perm)}
    return np.array([fold_of[d] for d in devices])


def lofo_scores(X_ai, fam_ai, X_re, dev_re, rng, permute=False, families=None):
    """Para cada familia f y cada pliegue de dispositivos g: entrena con IA != f y reales fuera de g; puntua IA f y reales de g."""
    fold = device_folds(dev_re, rng)
    fams = families or sorted(set(fam_ai))
    res = {f: {"ai": [], "re": [], "re_dev": []} for f in fams}
    for f in fams:
        for g in range(N_DEV_FOLDS):
            tr_ai, tr_re = fam_ai != f, fold != g
            Xtr = np.vstack([X_ai[tr_ai], X_re[tr_re]]); ytr = np.r_[np.ones(tr_ai.sum()), np.zeros(tr_re.sum())]
            if permute:
                ytr = rng.permutation(ytr)
            m = model().fit(Xtr, ytr)
            res[f]["re"] += list(m.decision_function(X_re[fold == g])); res[f]["re_dev"] += list(np.array(dev_re)[fold == g])
            if g == 0:
                pass
            res[f]["ai_g"] = res[f].get("ai_g", []) + [m.decision_function(X_ai[fam_ai == f])]
    for f in fams:  # cada IA de f se puntua con 5 modelos (uno por pliegue): se promedia por imagen
        res[f]["ai"] = list(np.mean(res[f].pop("ai_g"), axis=0))
    return res


def macro_from(res, fams, rng=None, cluster=False):
    vals = []
    for f in fams:
        ai = np.array(res[f]["ai"]); re = np.array(res[f]["re"]); dev = np.array(res[f]["re_dev"])
        if rng is not None:
            ai = rng.choice(ai, len(ai))
            if cluster:
                ud = sorted(set(dev)); pick = rng.choice(len(ud), len(ud)); re = np.concatenate([re[dev == ud[i]] for i in pick])
        vals.append(auc(ai, re))
    return float(np.mean(vals)), vals


def evaluate_features(name, X_ai, fam_ai, X_re, dev_re, base_ai, base_re, rng, n_boot, n_perm):
    fams = sorted(set(fam_ai))
    res = lofo_scores(X_ai, fam_ai, X_re, dev_re, np.random.default_rng(SEED))
    macro, per = macro_from(res, fams)
    boots = [macro_from(res, fams, rng, cluster=True)[0] for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    base = lofo_scores(base_ai, fam_ai, base_re, dev_re, np.random.default_rng(SEED))
    base_macro, _ = macro_from(base, fams)
    perm = [] if n_perm == 0 else [macro_from(lofo_scores(X_ai, fam_ai, X_re, dev_re, np.random.default_rng(SEED + 1 + i), permute=True), fams)[0] for i in range(n_perm)]
    out = {"macro_auc": macro, "ic95": [float(lo), float(hi)], "por_familia": {str(f): v for f, v in zip(fams, per)}, "linea_base_bytes": base_macro, "permutacion_media": float(np.mean(perm)) if perm else None, "permutacion_sd": float(np.std(perm)) if perm else None}
    LOG.info("L2 %-8s macro AUC LOFO=%.3f [%.3f, %.3f]; base bytes=%.3f; permutacion=%s", name, macro, lo, hi, base_macro, f"{np.mean(perm):.3f} (sd {np.std(perm):.3f})" if perm else "n/a")
    LOG.info("L2 %-8s por familia: %s", name, {str(f): round(v, 3) for f, v in zip(fams, per)})
    return out


def device_id_accuracy(X_re, dev_re):
    dev = np.array(dev_re); y = dev
    counts = collections.Counter(dev)
    keep = np.array([counts[d] >= N_DEV_FOLDS for d in dev])
    X, y = X_re[keep], y[keep]
    if len(set(y)) < 3:
        return None
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(N_DEV_FOLDS, shuffle=True, random_state=SEED).split(X, y):
        pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr]).predict(X[te])
    return float(np.mean(pred == y)), 1.0 / len(set(y))


def verdict(ev, devacc):
    if devacc is None:
        return "indeterminada"
    m, lo = ev["macro_auc"], ev["ic95"][0]
    ff = ev["por_familia"].get("firefly")  # control de formato: unica familia IA en JPEG, comparada con reales JPEG
    if m < 0.60 or lo <= 0.50:
        return "no apoyada"
    if ff is None:
        return "indeterminada"
    if m >= SUPPORT_AUC and lo > SUPPORT_LO and ff >= FORMAT_CTRL_MIN and devacc[0] < DEVID_MAX_ACC:
        return "apoyada"
    return "indeterminada"


def analyze_l2(recs, rng, n_boot, n_perm):
    reals = [r for r in recs if r["label"] == "real" and r["group"] == "real" and r["lf_eq"] and r["res_spec"]]
    ai = [r for r in recs if r["label"] == "fake" and r["lf_eq"] and r["res_spec"]]
    fam_ai = np.array([r["group"] for r in ai]); dev_re = [r["device"] for r in reals]
    def base(rs):
        return np.array([[np.log(r["bytes"] / (r["width"] * r["height"])), np.log(r["width"] * r["height"]), float(r["format"] == "JPEG")] for r in rs])
    out = {"n_real": len(reals), "n_ia": len(ai), "familias": sorted(str(f) for f in set(fam_ai))}
    LOG.info("L2: %d reales (%d dispositivos), %d IA, familias %s", len(reals), len(set(dev_re)), len(ai), out["familias"])
    for name, key in (("lf_eq", "lf_eq"), ("res_spec", "res_spec")):
        X_ai = np.array([r[key] for r in ai]); X_re = np.array([r[key] for r in reals])
        ev = evaluate_features(name, X_ai, fam_ai, X_re, dev_re, base(ai), base(reals), rng, n_boot, n_perm)
        devacc = device_id_accuracy(X_re, dev_re)
        ev["identificacion_dispositivo"] = None if devacc is None else {"exactitud": devacc[0], "azar": devacc[1]}
        LOG.info("L2 %-8s identificacion de dispositivo entre reales: %s", name, None if devacc is None else f"exactitud {devacc[0]:.3f} (azar {devacc[1]:.3f})")
        keep = fam_ai != "firefly"
        if keep.sum() and "firefly" in set(fam_ai):
            ev["sin_firefly"] = evaluate_features(name + "-sinFF", X_ai[keep], fam_ai[keep], X_re, dev_re, base(ai)[keep], base(reals), rng, max(n_boot // 4, 100), 0)
        ev["veredicto"] = verdict(ev, devacc)
        ev["etiqueta"] = "principal" if name == "lf_eq" else "secundario exploratorio (sin igualar procesado)"
        LOG.info("L2 %-8s VEREDICTO H1-L2: %s (%s)", name, ev["veredicto"], ev["etiqueta"])
        out[name] = ev
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir"); p.add_argument("--mapping-only", action="store_true"); p.add_argument("--self-test", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000); p.add_argument("--n-perm", type=int, default=20)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if not a.run_dir:
        p.error("falta --run-dir")
    manifest, recs = load(a.run_dir)
    table, status = build_mapping(recs, manifest["k_ids"])
    print_mapping(table, status, recs)
    if a.mapping_only:
        return
    rng = np.random.default_rng(SEED)
    result = {"mapeo": {d: {"estado": status[d], "k": table[d]} for d in table}}
    result["prnu"] = analyze_prnu(recs, manifest, table, status, rng, a.n_boot)
    result["l2"] = analyze_l2(recs, rng, a.n_boot, a.n_perm)
    with open(os.path.join(a.run_dir, "h1-analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.run_dir, "h1-analysis.json"))


def synthetic(signal, seed=3, n_dev=25, per=5, n_fam=4, per_fam=40, with_k=True):
    rng = np.random.default_rng(seed)
    k_ids = [f"vision_D{i:02d}_cam" for i in range(1, 11)] + ["dresden_Canon_Ixus55_0"]
    recs = []
    def rec(key, group, label, device, ai_shift):
        lf = rng.normal(0, 1, 32) + ai_shift * (label == "fake") * np.linspace(1, 0, 32)
        sp = rng.normal(0, 1, 64)
        pr = list(rng.normal(0, 0.004, len(k_ids)))
        return {"key": key, "group": group, "label": label, "rel": key, "device": device, "status": "ok", "format": "JPEG" if (label == "real" or group == "firefly") else "PNG",
                "width": 3000, "height": 2000, "bytes": int(rng.integers(2e6, 3e6)), "lf_eq": lf.tolist(), "lf_raw": lf.tolist(), "res_spec": sp.tolist(), "prnu": pr}
    for d in range(n_dev):
        dev = f"vision_D{d + 1:02d}_cam" if d < 10 else f"unknown_dev{d}"
        shift = rng.normal(0, 0.05, 32)
        for i in range(per):
            r = rec(f"real/{dev}/{i}", "real", "real", dev, 0)
            r["lf_eq"] = (np.array(r["lf_eq"]) + shift).tolist()
            if d < 10 and with_k:
                r["prnu"][d] = 0.05 + rng.normal(0, 0.01)
            recs.append(r)
    for f in range(n_fam):
        name = "firefly" if f == n_fam - 1 else f"fam{f}"
        for i in range(per_fam):
            recs.append(rec(f"fake/f{f}/{i}", name, "fake", None, signal))
    return {"k_ids": k_ids}, recs


def self_test():
    rng = np.random.default_rng(1)
    man, recs = synthetic(signal=1.5)
    table, status = build_mapping(recs, man["k_ids"])
    assert status["vision_D03_cam"] == "mapeada" and table["vision_D03_cam"] == ["vision_D03_cam"] and status["unknown_dev20"] == "ausente"
    assert device_to_entry("dresden_Canon_Ixus55_0", man["k_ids"]) == ["dresden_Canon_Ixus55_0"] and device_to_entry("dresden_Nikon_D70_0", man["k_ids"]) == []
    assert device_to_entry("modern_C05", ["modern_C05", "vision_D05_x"]) == ["modern_C05"] and device_to_entry("D03_Huawei_P9", ["vision_D03_Huawei_P9"]) == ["vision_D03_Huawei_P9"]
    assert len(device_to_entry("D03_x", ["vision_D03_a", "x_D03_b"])) == 2
    p = analyze_prnu(recs, man, table, status, rng, 200)
    assert p["P0"]["valido"] and p["P0"]["auc_propia_vs_ajenas"] > 0.95 and p["P0"]["top1"] > 0.95
    assert p["P1"]["real_con_K"]["auc_real_alto_vs_ia"] > 0.9 and 0.3 < p["P1"]["real_camara_ausente"]["auc_real_alto_vs_ia"] < 0.7
    man2, recs2 = synthetic(signal=1.5, with_k=False)
    t2, s2 = build_mapping(recs2, man2["k_ids"])
    assert not analyze_prnu(recs2, man2, t2, s2, rng, 50)["P0"]["valido"]
    pos = analyze_l2(recs, rng, 100, 3)
    assert pos["lf_eq"]["macro_auc"] > 0.85 and pos["lf_eq"]["veredicto"] == "apoyada", pos["lf_eq"]
    assert abs(pos["lf_eq"]["permutacion_media"] - 0.5) < 0.1 and 0.3 < pos["res_spec"]["macro_auc"] < 0.7 and pos["lf_eq"]["por_familia"]["firefly"] > 0.9
    _, recs0 = synthetic(signal=0.0)
    neg = analyze_l2(recs0, rng, 100, 3)
    assert neg["lf_eq"]["veredicto"] == "no apoyada" and 0.35 < neg["lf_eq"]["macro_auc"] < 0.65, neg["lf_eq"]
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "manifest.json"), "w") as f: json.dump(man, f)
    with open(os.path.join(d, "h1-features.jsonl"), "w") as f: f.write("\n".join(json.dumps(r) for r in recs))
    m, rs = load(d); assert len(rs) == len(recs)
    LOG.info("autotest OK (12 comprobaciones)")


if __name__ == "__main__":
    main()
