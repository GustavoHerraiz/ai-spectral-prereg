#!/usr/bin/env python3
"""Analisis posterior de H1 (EXPLORATORIO salvo la parte `controles`, fijada en PREREGISTRO_H1.md seccion 7 antes de ver datos de control).

Con h1-extract-run1 (ya existente):
  geometria : alineacion del recorte 512 con la rejilla 8x8 y tamano, por grupo.
  firefly   : que K gana el max NCC en firefly y de que tipo (diagnostico del 56 % sobre el umbral).
  mismo_formato : reales JPEG frente a firefly JPEG (unica comparacion con formato igual), validacion por dispositivo.
  bins      : AUC univariado por bin de frecuencia (donde esta la senal).
Con --controls-dir (salida de h1-controls-extract.py):
  controles : L2 repetido en cada brazo (mod8, q75, q92, down, down_q92) con la misma validacion y criterio.
Uso: python3 h1-post.py --run-dir DIR [--controls-dir DIR] [--n-boot 300]    |    python3 h1-post.py --self-test
"""
import argparse, collections, importlib.util, json, logging, os, sys
import numpy as np

LOG = logging.getLogger("h1-post")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


ANA = load_module("h1analyze", os.path.join(HERE, "h1-analyze.py"))
SEED, THRESH = ANA.SEED, ANA.THRESH
SPEC_EDGES = np.logspace(np.log10(0.004), np.log10(0.5), 65)
LF_EDGES = np.linspace(0.02, 0.35, 33)
centers = lambda e, geo: np.sqrt(e[:-1] * e[1:]) if geo else (e[:-1] + e[1:]) / 2


def geometry(recs):
    out = {}
    groups = collections.defaultdict(list)
    for r in recs:
        if r["group"] != "real-dresden":
            groups[r["group"]].append(r)
    for g, rs in sorted(groups.items()):
        al = [((r["width"] - 512) // 2 % 8 == 0) and ((r["height"] - 512) // 2 % 8 == 0) for r in rs]
        short = [min(r["width"], r["height"]) for r in rs]
        out[g] = {"n": len(rs), "frac_recorte_alineado_8": float(np.mean(al)), "lado_menor_mediana": float(np.median(short)), "formato": dict(collections.Counter(r["format"] for r in rs))}
        LOG.info("GEOMETRIA %-20s n=%-4d recorte alineado a 8: %.2f; lado menor mediana %d; formato %s", g, len(rs), out[g]["frac_recorte_alineado_8"], np.median(short), out[g]["formato"])
    return out


def firefly_diag(manifest, recs):
    ids = manifest["k_ids"]
    ff = [r for r in recs if r["group"] == "firefly" and r["prnu"]]
    if not ff:
        return None
    best = [int(np.argmax(r["prnu"])) for r in ff]
    smax = np.array([max(r["prnu"]) for r in ff])
    cnt = collections.Counter(ids[b] for b in best)
    over = collections.Counter(ids[b].split("_")[0] for b, s in zip(best, smax) if s >= THRESH)
    out = {"n": len(ff), "frac_sobre_umbral": float(np.mean(smax >= THRESH)), "k_distintas_ganadoras": len(cnt), "top_k": cnt.most_common(6), "sobre_umbral_por_tipo_K": dict(over),
           "max_ncc_percentiles_5_50_95": [float(x) for x in np.percentile(smax, [5, 50, 95])]}
    LOG.info("FIREFLY max NCC: %d imagenes, %.2f sobre umbral, %d K distintas ganadoras; top: %s", len(ff), out["frac_sobre_umbral"], len(cnt), cnt.most_common(6))
    LOG.info("FIREFLY sobre umbral por tipo de K: %s; percentiles 5/50/95 del max NCC: %s", dict(over), [round(x, 4) for x in out["max_ncc_percentiles_5_50_95"]])
    return out


def within_format(X_re, dev_re, X_ff, n_boot, seed=SEED):
    rng = np.random.default_rng(seed)
    fold_re = ANA.device_folds(dev_re, np.random.default_rng(SEED))
    fold_ff = rng.permutation(len(X_ff)) % ANA.N_DEV_FOLDS
    s_re, s_ff = np.empty(len(X_re)), np.empty(len(X_ff))
    for g in range(ANA.N_DEV_FOLDS):
        Xtr = np.vstack([X_ff[fold_ff != g], X_re[fold_re != g]]); ytr = np.r_[np.ones((fold_ff != g).sum()), np.zeros((fold_re != g).sum())]
        m = ANA.model().fit(Xtr, ytr)
        s_re[fold_re == g] = m.decision_function(X_re[fold_re == g]); s_ff[fold_ff == g] = m.decision_function(X_ff[fold_ff == g])
    dev = np.array(dev_re); ud = sorted(set(dev)); idx = {d: np.where(dev == d)[0] for d in ud}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(ud), len(ud))
        boots.append(ANA.auc(rng.choice(s_ff, len(s_ff)), s_re[np.concatenate([idx[ud[i]] for i in pick])]))
    return {"auc": ANA.auc(s_ff, s_re), "ic95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}


def split_real_ai(recs):
    reals = [r for r in recs if r["label"] == "real" and r["group"] == "real" and r["lf_eq"] and r["res_spec"]]
    ai = [r for r in recs if r["label"] == "fake" and r["lf_eq"] and r["res_spec"]]
    return reals, ai


def mismo_formato(recs, n_boot):
    reals, ai = split_real_ai(recs)
    ff = [r for r in ai if r["group"] == "firefly"]
    if not ff:
        return None
    dev = [r["device"] for r in reals]
    base = lambda rs: np.array([[np.log(r["bytes"] / (r["width"] * r["height"])), np.log(r["width"] * r["height"])] for r in rs])
    out = {"n_real": len(reals), "n_firefly": len(ff)}
    for name, fn in (("lf_eq", lambda rs: np.array([r["lf_eq"] for r in rs])), ("res_spec", lambda rs: np.array([r["res_spec"] for r in rs])), ("bytes_px_tamano", base)):
        out[name] = within_format(fn(reals), dev, fn(ff), n_boot)
        LOG.info("MISMO FORMATO (JPEG real vs firefly) %-16s AUC=%.3f [%.3f, %.3f]", name, out[name]["auc"], *out[name]["ic95"])
    return out


def bins(recs):
    reals, ai = split_real_ai(recs)
    out = {}
    for feat, edges, geo in (("res_spec", SPEC_EDGES, True), ("lf_eq", LF_EDGES, False)):
        c = centers(edges, geo); Xr = np.array([r[feat] for r in reals])
        for name, sel in (("IA_PNG", [r for r in ai if r["group"] != "firefly"]), ("firefly", [r for r in ai if r["group"] == "firefly"])):
            if not sel:
                continue
            Xa = np.array([r[feat] for r in sel]); a = np.array([ANA.auc(Xa[:, j], Xr[:, j]) for j in range(Xa.shape[1])])
            top = np.argsort(-np.abs(a - 0.5))[:6]
            out[f"{feat}_{name}"] = [{"freq": float(c[j]), "auc_ia_alta": float(a[j])} for j in top]
            LOG.info("BINS %-8s %-7s mayor separacion (ciclos/px: AUC IA>real): %s", feat, name, [(round(float(c[j]), 4), round(float(a[j]), 2)) for j in top])
    return out


def control_verdict(arm_results):
    out = {}
    for feat in ("lf_eq", "res_spec"):
        ok = lambda ev: ev["macro_auc"] >= ANA.SUPPORT_AUC and ev["ic95"][0] > ANA.SUPPORT_LO and ev["por_familia"].get("firefly", 0) >= ANA.FORMAT_CTRL_MIN
        bad = lambda ev: ev["macro_auc"] < 0.60 or ev["ic95"][0] <= 0.50
        evs = {a: arm_results[a][feat] for a in ("mod8", "q75", "q92") if a in arm_results}
        if len(evs) < 3:
            out[feat] = "incompleto"; continue
        out[feat] = "sobrevive" if all(ok(e) for e in evs.values()) else "colapsa" if bad(evs["q75"]) or bad(evs["q92"]) else "parcial"
        eq_ok = all(arm_results[a][feat]["linea_base_bytes"] < 0.80 for a in ("q75", "q92"))
        out[feat + "_igualacion_efectiva"] = bool(eq_ok)
        LOG.info("CONTROL VEREDICTO %-8s: %s (bytes recomprimidos < 0.80 en q75/q92: %s)", feat, out[feat], eq_ok)
    return out


def controls(controls_dir, n_boot):
    recs = []
    with open(os.path.join(controls_dir, "h1-controls.jsonl"), encoding="utf-8") as f:
        for l in f:
            if l.strip():
                recs.append(json.loads(l))
    ok = [r for r in recs if r["status"] == "ok"]
    LOG.info("controles: %d ok, %d con error", len(ok), len(recs) - len(ok))
    out = {}
    for arm in ok[0]["arms"]:
        arm_recs = [{"key": r["key"], "group": r["group"], "label": r["label"], "device": r["device"], **{k: r["arms"][arm][k] for k in ("lf_eq", "res_spec", "bytes", "width", "height", "format")}} for r in ok]
        LOG.info("---- brazo %s", arm)
        res = ANA.analyze_l2(arm_recs, np.random.default_rng(SEED), n_boot, 0)
        reals, ai = split_real_ai(arm_recs)
        ff = [r for r in ai if r["group"] == "firefly"]
        dev = [r["device"] for r in reals]
        if ff:
            for feat in ("lf_eq", "res_spec"):
                res[feat]["mismo_formato_firefly"] = within_format(np.array([r[feat] for r in reals]), dev, np.array([r[feat] for r in ff]), n_boot)
                LOG.info("CONTROL %-8s %-8s JPEG real vs firefly AUC=%.3f [%.3f, %.3f]", arm, feat, res[feat]["mismo_formato_firefly"]["auc"], *res[feat]["mismo_formato_firefly"]["ic95"])
        out[arm] = res
    out["veredicto"] = control_verdict(out)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir"); p.add_argument("--controls-dir"); p.add_argument("--n-boot", type=int, default=300); p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if not a.run_dir:
        p.error("falta --run-dir")
    manifest, recs = ANA.load(a.run_dir)
    result = {"geometria": geometry(recs), "firefly": firefly_diag(manifest, recs), "mismo_formato": mismo_formato(recs, a.n_boot), "bins": bins(recs)}
    if a.controls_dir:
        result["controles"] = controls(a.controls_dir, a.n_boot)
    path = os.path.join(a.controls_dir or a.run_dir, "h1-post.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    LOG.info("escrito %s", path)


def self_test():
    import tempfile
    man, recs = ANA.synthetic(signal=1.5)
    g = geometry(recs)
    assert g["real"]["n"] == 125 and g["firefly"]["formato"] == {"PNG": 40} or g["firefly"]["formato"] == {"JPEG": 40}
    for r in recs:
        if r["group"] == "firefly":
            r["prnu"][2] = 0.03
    fd = firefly_diag(man, recs)
    assert fd["frac_sobre_umbral"] == 1.0 and fd["top_k"][0][0] == man["k_ids"][2]
    mf = mismo_formato(recs, 50)
    assert mf["lf_eq"]["auc"] > 0.85 and 0.3 < mf["res_spec"]["auc"] < 0.7, mf
    b = bins(recs)
    assert b["lf_eq_firefly"][0]["auc_ia_alta"] > 0.6 and len(b["res_spec_IA_PNG"]) == 6
    # controles sinteticos: la senal sobrevive en todos los brazos salvo en q75/q92, donde se anula
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(5)
    with open(os.path.join(d, "h1-controls.jsonl"), "w") as f:
        for r in recs:
            arms = {}
            for arm in ("mod8", "q75", "q92", "down", "down_q92"):
                lf = np.array(r["lf_eq"]) if arm in ("mod8", "down") else rng.normal(0, 1, 32)
                arms[arm] = {"lf_eq": lf.tolist(), "res_spec": r["res_spec"], "bytes": 100000, "width": 512, "height": 512, "format": "JPEG"}
            f.write(json.dumps({"key": r["key"], "group": r["group"], "label": r["label"], "device": r["device"], "status": "ok", "arms": arms}) + "\n")
    c = controls(d, 40)
    assert c["veredicto"]["lf_eq"] == "colapsa" and c["mod8"]["lf_eq"]["macro_auc"] > 0.85, c["veredicto"]
    LOG.info("autotest OK (6 comprobaciones)")


if __name__ == "__main__":
    main()
