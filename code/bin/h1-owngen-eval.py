#!/usr/bin/env python3
"""H1-L2: evaluacion de IA propia (historial conocido) contra reales ya extraidos. Ver PREREGISTRO_H1.md seccion 19.

Entrena con las 700 IA Synthbuster + reales de un corpus (run3 o dresden) y puntua IA propia + reales del OTRO corpus (2 pares).
Features de reales y de Synthbuster: h1-resample-feats.jsonl (mismas cadenas). IA propia: PNG 512 -> mismas cadenas A0..A6 (h1-resample.process).
Brazos: primario A5/R1 (IA con paso bajo + q90 + q92 frente a reales q92) ; sensibilidad A0/R0 y A1/R1.
AUC (IA propia > reales del corpus de test) por generador, IC95 por bootstrap (imagenes IA; dispositivos de reales).
Uso:
  python3 h1-owngen-eval.py --own-dir DIR --resample-feats FILE --out OUT [--workers 2]
  python3 h1-owngen-eval.py --self-test
"""
import argparse, json, logging, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image

LOG = logging.getLogger("h1-owngen-eval")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


RS = load_module("h1resample", os.path.join(HERE, "h1-resample.py"))
LOCO, ANA, NAT = RS.LOCO, RS.LOCO.ANA, RS.NAT
FEATS, PAIRS = LOCO.FEATS, NAT.PAIRS
ARMS = {"primario A5/R1": ("A5", "R1"), "A0/R0": ("A0", "R0"), "A1/R1": ("A1", "R1")}
PRIMARY = "primario A5/R1"


def own_items(own_dir):
    items = []
    for fn in sorted(os.listdir(own_dir)):
        if fn.lower().endswith(".png"):
            gen = fn.split("_")[0]
            items.append((os.path.join(own_dir, fn), {"key": "own/" + fn, "corpus": "ai", "group": gen, "device": None}))
    return items


def extract_own(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h1-owngen-feats.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8") if l.strip()}
    items = [it for it in own_items(a.own_dir) if it[1]["key"] not in done]
    for p, _ in items:
        with Image.open(p) as im:
            if im.size != (512, 512):
                raise ValueError(f"{p}: {im.size}, se esperaba 512x512 nativo")
    LOG.info("IA propia: %d por hacer (%d ya hechas)", len(items), len(done))
    with open(path, "a", encoding="utf-8") as out, Pool(a.workers) as pool:
        for rec in pool.imap_unordered(RS.process, items, chunksize=2):
            out.write(json.dumps(rec) + "\n"); out.flush()
    return path


def boot_auc(s_ai, s_re, dev_re, n_boot, rng):
    ud = sorted(set(dev_re)); idx = {d: np.where(np.asarray(dev_re) == d)[0] for d in ud}; v = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[ud[i]] for i in rng.choice(len(ud), len(ud))])
        v.append(ANA.auc(rng.choice(s_ai, len(s_ai)), s_re[sel]))
    return LOCO.ci(v)


def analyze(a, own_path):
    rs = [json.loads(l) for l in open(a.resample_feats, encoding="utf-8") if l.strip()]
    rs = [r for r in rs if r["status"] == "ok"]
    own = [json.loads(l) for l in open(own_path, encoding="utf-8") if l.strip()]
    own = [r for r in own if r["status"] == "ok"]
    syn = [r for r in rs if r["corpus"] == "ai"]; reals = [r for r in rs if r["corpus"] != "ai"]
    gens = sorted({r["group"] for r in own})
    LOG.info("Synthbuster %d; reales %s; IA propia %s", len(syn), {c: sum(r["corpus"] == c for r in reals) for c in ("run3", "dresden")}, {g: sum(r["group"] == g for r in own) for g in gens})
    corp = np.array([r["corpus"] for r in reals]); dev = np.array([r["device"] for r in reals])
    out = {}
    LOG.info("| rasgo | brazo | generador | run3->dresden | dresden->run3 | media | veredicto |")
    for feat in FEATS:
        out[feat] = {}
        for arm, (aa, ra) in ARMS.items():
            X_syn = np.array([r["arms"][aa][feat] for r in syn]); X_re = np.array([r["arms"][ra][feat] for r in reals])
            for g in gens:
                X_own = np.array([r["arms"][aa][feat] for r in own if r["group"] == g]); res = {}
                for x, y in PAIRS:
                    trn, tst = corp == x, corp == y
                    m = ANA.model().fit(np.vstack([X_syn, X_re[trn]]), np.r_[np.ones(len(X_syn)), np.zeros(trn.sum())])
                    s_own, s_re = m.decision_function(X_own), m.decision_function(X_re[tst])
                    res[f"{x}->{y}"] = {"auc": ANA.auc(s_own, s_re), "ic95": boot_auc(s_own, s_re, dev[tst], a.n_boot, np.random.default_rng(ANA.SEED)), "n_own": len(X_own)}
                v, mean, lo = LOCO.verdict({"cruzado": res})
                v = {"aguanta": "se confirma", "artefacto de corpus": "propiedad de Synthbuster"}.get(v, "indeterminada")
                out[feat][f"{arm}|{g}"] = {"pares": res, "media": mean, "min_ic_inf": lo, "veredicto": v}
                LOG.info("| %-8s | %-14s | %-8s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f | %s |", feat, arm, g, *[z for p in res.values() for z in (p["auc"], *p["ic95"])], mean, v)
        LOG.info("VEREDICTO H1-L2-IAPROPIA %s (primario, sdturbo): %s", feat, out[feat].get(f"{PRIMARY}|sdturbo", {}).get("veredicto", "sin sdturbo"))
    with open(os.path.join(a.out, "h1-owngen.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h1-owngen.json"))
    return out


def self_test():
    import shutil
    rng = np.random.default_rng(21)
    tmp = tempfile.mkdtemp()
    try:
        feats = lambda mean: {"res_spec": (rng.normal(mean, 1, 64)).tolist(), "lf_eq": (rng.normal(0, 1, 32)).tolist()}
        def rec(corpus, group, dev, arms, mean):
            return {"key": f"{corpus}{group}{rng.integers(1e9)}", "corpus": corpus, "group": group, "device": dev, "status": "ok", "arms": {k: feats(mean) for k in arms}}
        def build(own_mean):
            rows = [rec(c, "real", f"{c}{d}", ("R0", "R1"), 0.0) for c in ("run3", "dresden") for d in range(10) for _ in range(3)]
            rows += [rec("ai", f"fam{f}", None, ("A0", "A1", "A5"), 0.8) for f in range(4) for _ in range(25)]
            rf = os.path.join(tmp, f"rs{own_mean}.jsonl"); open(rf, "w").write("\n".join(json.dumps(r) for r in rows))
            of = os.path.join(tmp, f"own{own_mean}.jsonl")
            open(of, "w").write("\n".join(json.dumps(rec("ai", "sdturbo", None, ("A0", "A1", "A5"), own_mean)) for _ in range(30)))
            return argparse.Namespace(resample_feats=rf, out=tmp, n_boot=20), of
        a, of = build(0.8); assert analyze(a, of)["res_spec"][f"{PRIMARY}|sdturbo"]["veredicto"] == "se confirma", "positivo"
        a, of = build(0.0); r = analyze(a, of)["res_spec"][f"{PRIMARY}|sdturbo"]; assert r["veredicto"] == "propiedad de Synthbuster", r["media"]
        d = os.path.join(tmp, "bad"); os.makedirs(d); Image.new("RGB", (256, 256)).save(os.path.join(d, "sdturbo_000.png"))
        try:
            extract_own(argparse.Namespace(own_dir=d, out=os.path.join(tmp, "o"), workers=1)); raise SystemExit("debio rechazar tamano")
        except ValueError:
            pass
        LOG.info("autotest OK (3 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("own-dir", "resample-feats", "out"):
        p.add_argument("--" + k)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("own_dir", "resample_feats", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    analyze(a, extract_own(a))


if __name__ == "__main__":
    main()
