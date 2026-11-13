#!/usr/bin/env python3
"""H2 etapa 1: replica y dependencia de los pasos del desplazamiento de sd-turbo sin cadena (A1). Ver PREREGISTRO_H2.md seccion 4.

Condiciones (mismas semillas y prompts, h1-owngen-make.py --seed-offset 200000 --new-prompts): E1 = 2 pasos, E2a = 1 paso, E2b = 4 pasos.
Estadistico S = AUC de A1 frente a R1 con el modelo Synthbuster A5 vs R1 completo (X0 de h2-sdfamily), media run3<->dresden.
Uso:
  python3 h2-steps.py --e1-dir D2 --s1-dir D1 --s4-dir D4 --resample-feats F --out OUT [--workers 2]
  python3 h2-steps.py --self-test
"""
import argparse, json, logging, os, sys, tempfile
import numpy as np
import importlib.util

LOG = logging.getLogger("h2-steps")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


SD = load_module("h2sdfamily", os.path.join(HERE, "h2-sdfamily.py"))
CH, EV, ANA, LOCO = SD.CH, SD.EV, SD.ANA, SD.LOCO
COND = {"E1_2pasos": 2, "E2a_1paso": 1, "E2b_4pasos": 4}
DIFF_MIN = 0.15


def read_own(path):
    rows = [r for r in CH.read(path) if r["group"] == SD.GEN and r.get("status", "ok") == "ok"]
    return sorted(rows, key=lambda r: r["key"])


def paired_diff(syn, reals, corp, dev, o1, o4, n_boot):
    """Diferencia S(1 paso) - S(4 pasos): remuestreo conjunto de indices de imagen (emparejadas por semilla) y de dispositivos."""
    rng = np.random.default_rng(ANA.SEED); X_syn = np.array([r["arms"]["A5"][SD.FEAT] for r in syn])
    X_r1 = np.array([r["arms"]["R1"][SD.FEAT] for r in reals])
    A, B = np.array([r["arms"]["A1"][SD.FEAT] for r in o1]), np.array([r["arms"]["A1"][SD.FEAT] for r in o4])
    parts = []
    for x, y in CH.PAIRS:
        tx, ty = corp == x, corp == y
        m = CH.fit(X_syn, X_r1[tx]); sr = m.decision_function(X_r1[ty]); d = dev[ty]
        ud = sorted(set(d)); ix = {k: np.where(d == k)[0] for k in ud}
        parts.append((m.decision_function(A), m.decision_function(B), sr, ud, ix))
    point = float(np.mean([ANA.auc(a, s) - ANA.auc(b, s) for a, b, s, _, _ in parts]))
    vals = []
    for _ in range(n_boot):
        v = []
        for a, b, s, ud, ix in parts:
            sel = np.concatenate([ix[ud[i]] for i in rng.choice(len(ud), len(ud))]); k = rng.choice(len(a), len(a))
            v.append(ANA.auc(a[k], s[sel]) - ANA.auc(b[k], s[sel]))
        vals.append(np.mean(v))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, [float(lo), float(hi)]


def analyze(a):
    rs = CH.read(a.resample_feats)
    syn = [r for r in rs if r["corpus"] == "ai"]; reals = [r for r in rs if r["corpus"] != "ai"]
    corp = np.array([r["corpus"] for r in reals]); dev = np.array([r["device"] for r in reals])
    own = {c: read_own(p) for c, p in (("E1_2pasos", a.e1_feats), ("E2a_1paso", a.s1_feats), ("E2b_4pasos", a.s4_feats))}
    for c, rows in own.items():
        LOG.info("%s: %d imagenes", c, len(rows))
    out = {"condiciones": {}}
    LOG.info("| condicion | n | S media | mín. IC inf. | veredicto base |")
    for c, rows in own.items():
        r = SD.score(syn, reals, corp, dev, rows, (), a.n_boot); out["condiciones"][c] = {**r, "n": len(rows), "pasos": COND[c]}
        LOG.info("| %-10s | %d | %.3f | %.3f | %s |", c, len(rows), r["media"], r["min_ic_inf"], r["veredicto_base"])
    e1 = out["condiciones"]["E1_2pasos"]
    if e1["media"] >= LOCO.AGUANTA_MEAN and e1["min_ic_inf"] >= LOCO.AGUANTA_LO:
        rep = "se reproduce"
    elif e1["media"] < LOCO.ARTEFACTO_MEAN:
        rep = "no se reproduce"
    else:
        rep = "indeterminada"
    out["replica"] = rep; out["pasos"] = "no evaluada (la replica no se reproduce)"
    if rep == "se reproduce":
        o1, o4 = own["E2a_1paso"], own["E2b_4pasos"]
        if len(o1) != len(o4):
            raise ValueError("las condiciones de 1 y 4 pasos deben tener el mismo numero de imagenes (emparejadas)")
        diff, ci = paired_diff(syn, reals, corp, dev, o1, o4, a.n_boot)
        dep = abs(diff) >= DIFF_MIN and (ci[0] > 0 or ci[1] < 0)
        out["diferencia_1_menos_4"] = {"valor": diff, "ic95": ci}
        out["pasos"] = "depende de los pasos" if dep else "no depende de los pasos"
        LOG.info("S(1) - S(4) = %.3f IC95 [%.3f, %.3f]", diff, ci[0], ci[1])
    LOG.info("VEREDICTO H2-E1: replica %s; pasos: %s", out["replica"], out["pasos"])
    with open(os.path.join(a.out, "h2-steps.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h2-steps.json"))
    return out


def extract(a):
    paths = {}
    for tag, d in (("e1", a.e1_dir), ("s1", a.s1_dir), ("s4", a.s4_dir)):
        sub = os.path.join(a.out, "feats-" + tag)
        paths[tag] = EVAL.extract_own(argparse.Namespace(own_dir=d, out=sub, workers=a.workers))
    return paths


def self_test():
    import shutil
    rng = np.random.default_rng(5); tmp = tempfile.mkdtemp()
    try:
        def feats(sa):
            v = rng.normal(0, 1, 64); v[:32] += sa
            return {"res_spec": v.tolist(), "lf_eq": rng.normal(0, 1, 32).tolist()}
        rows = []
        for c in ("run3", "dresden"):
            for d in range(10):
                for i in range(3):
                    rows.append({"key": f"{c}/{d}/{i}", "corpus": c, "group": "real", "device": f"{c}{d}", "status": "ok", "arms": {"R1": feats(0)}})
        for f in SD.SD + SD.NON_SD:
            for i in range(25):
                rows.append({"key": f"ai/{f}/{i}", "corpus": "ai", "group": f, "device": None, "status": "ok", "arms": {"A5": feats(1.5)}})

        def w(name, rs):
            p = os.path.join(tmp, name)
            with open(p, "w") as fh:
                fh.write("\n".join(json.dumps(x) for x in rs))
            return p

        def own(sa):
            return [{"key": f"own/sdturbo_{i:03d}.png", "corpus": "ai", "group": SD.GEN, "device": None, "status": "ok", "arms": {"A1": feats(sa)}} for i in range(30)]

        def run(tag, s2, s1, s4):
            a = argparse.Namespace(resample_feats=w("rs.jsonl", rows), e1_feats=w(tag + "e1", own(s2)), s1_feats=w(tag + "s1", own(s1)), s4_feats=w(tag + "s4", own(s4)), out=tmp, n_boot=30)
            return analyze(a)
        r = run("dep", 1.5, 1.5, 0.0)
        assert r["replica"] == "se reproduce" and r["pasos"] == "depende de los pasos", r
        r = run("nodep", 1.5, 1.5, 1.5)
        assert r["replica"] == "se reproduce" and r["pasos"] == "no depende de los pasos", r
        r = run("norep", 0.0, 0.0, 0.0)
        assert r["replica"] == "no se reproduce" and r["pasos"].startswith("no evaluada"), r
        LOG.info("autotest OK (6 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("e1-dir", "s1-dir", "s4-dir", "resample-feats", "out"):
        p.add_argument("--" + k)
    p.add_argument("--n-boot", type=int, default=1000); p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("e1_dir", "s1_dir", "s4_dir", "resample_feats", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    os.makedirs(a.out, exist_ok=True)
    global EVAL
    EVAL = load_module("h1owngeneval", os.path.join(HERE, "h1-owngen-eval.py"))
    paths = extract(a)
    a.e1_feats, a.s1_feats, a.s4_feats = paths["e1"], paths["s1"], paths["s4"]
    analyze(a)


if __name__ == "__main__":
    main()
