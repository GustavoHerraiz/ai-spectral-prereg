#!/usr/bin/env python3
"""H2 etapa 0: el desplazamiento de sd-turbo sin cadena (A1) depende del linaje Stable Diffusion del entrenamiento? Ver PREREGISTRO_H2.md.

No genera imagenes: reutiliza los rasgos archivados (h1-resample-feats.jsonl: Synthbuster A5 y reales R1; h1-owngen-feats.jsonl: sd-turbo A1).
Modelo: Synthbuster A5 (con familias excluidas) vs R1(X); prueba: AUC de sd-turbo A1 vs R1(Y), run3<->dresden, rasgo res_spec.
Analisis: X0 sin exclusion (debe reproducir ~0,9 de la nota 5.13); X1 sin SD2; X2 sin SD1+SD2; X3 (primario) sin SD1+SD2+SDXL;
controles: excluir 3 de las 4 familias no SD (dalle2, dalle3, firefly, midjourney), mismo numero de familias que X3.
Uso:
  python3 h2-sdfamily.py --resample-feats F --own-feats F --out OUT
  python3 h2-sdfamily.py --self-test
"""
import argparse, itertools, json, logging, os, sys, tempfile
import numpy as np
import importlib.util

LOG = logging.getLogger("h2-sdfamily")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


CH = load_module("h1chain", os.path.join(HERE, "h1-chain.py"))
EV, ANA, LOCO = CH.EV, CH.ANA, CH.LOCO
FEAT = "res_spec"
SD = ("sd1", "stable-diffusion-2", "stable-diffusion-xl")
NON_SD = ("dalle2", "dalle3", "firefly", "midjourney")
GEN = "sdturbo"
EXCLUSIONES = {"X0_ninguna": (), "X1_sin_sd2": ("stable-diffusion-2",), "X2_sin_sd1_sd2": ("sd1", "stable-diffusion-2"), "X3_sin_linaje_sd": SD}
LABELS = {"aguanta": "no explicada", "artefacto de corpus": "explicada"}


def score(syn, reals, corp, dev, own, excl, n_boot):
    keep = [r for r in syn if r["group"] not in excl]
    X_syn = np.array([r["arms"]["A5"][FEAT] for r in keep])
    X_r1 = np.array([r["arms"]["R1"][FEAT] for r in reals]); O1 = np.array([r["arms"]["A1"][FEAT] for r in own])
    res = {}
    for x, y in CH.PAIRS:
        tx, ty = corp == x, corp == y
        m = CH.fit(X_syn, X_r1[tx])
        s, sr = m.decision_function(O1), m.decision_function(X_r1[ty])
        res[f"{x}->{y}"] = {"auc": ANA.auc(s, sr), "ic95": EV.boot_auc(s, sr, dev[ty], n_boot, np.random.default_rng(ANA.SEED))}
    v, mean, lo = LOCO.verdict({"cruzado": res})
    return {"pares": res, "media": mean, "min_ic_inf": lo, "veredicto_base": v, "n_syn_entrenamiento": len(keep)}


def analyze(a):
    rs = CH.read(a.resample_feats)
    own = [r for r in CH.read(a.own_feats) if r["group"] == GEN]
    syn = [r for r in rs if r["corpus"] == "ai"]
    reals = [r for r in rs if r["corpus"] != "ai"]
    corp = np.array([r["corpus"] for r in reals]); dev = np.array([r["device"] for r in reals])
    LOG.info("sintetico %d, reales %d, %s %d", len(syn), len(reals), GEN, len(own))
    out = {"exclusiones": {}, "controles": {}}
    LOG.info("| analisis | excluye | media | min IC inf | veredicto base |")
    for name, excl in EXCLUSIONES.items():
        r = score(syn, reals, corp, dev, own, excl, a.n_boot); out["exclusiones"][name] = {**r, "excluye": list(excl)}
        LOG.info("| %-18s | %-40s | %.3f | %.3f | %s |", name, ",".join(excl) or "-", r["media"], r["min_ic_inf"], r["veredicto_base"])
    ctrl = []
    for keep_one in NON_SD:
        excl = tuple(f for f in NON_SD if f != keep_one)
        r = score(syn, reals, corp, dev, own, excl, a.n_boot); out["controles"]["_".join(excl)] = {**r, "excluye": list(excl)}
        ctrl.append(r["media"])
        LOG.info("| control            | %-40s | %.3f | %.3f | %s |", ",".join(excl), r["media"], r["min_ic_inf"], r["veredicto_base"])
    prim = out["exclusiones"]["X3_sin_linaje_sd"]; cmean = float(np.mean(ctrl))
    if prim["media"] < LOCO.ARTEFACTO_MEAN and cmean >= LOCO.AGUANTA_MEAN:
        v = "explicada por linaje SD del entrenamiento"
    elif prim["media"] < LOCO.ARTEFACTO_MEAN:
        v = "indeterminada: la caida tambien ocurre al excluir familias no SD (perdida generica de entrenamiento)"
    elif prim["media"] >= LOCO.AGUANTA_MEAN and prim["min_ic_inf"] >= LOCO.AGUANTA_LO:
        v = "no explicada por linaje SD"
    else:
        v = "indeterminada"
    out["media_controles"] = cmean; out["veredicto"] = v
    LOG.info("VEREDICTO H2-E0 (primario X3=%.3f, media controles=%.3f): %s", prim["media"], cmean, v)
    with open(os.path.join(a.out, "h2-sdfamily.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h2-sdfamily.json"))
    return out


def self_test():
    import shutil
    rng = np.random.default_rng(11)
    tmp = tempfile.mkdtemp()
    try:
        def feats(sa, sb):
            v = rng.normal(0, 1, 64); v[:32] += sa; v[32:] += sb
            return {"res_spec": v.tolist(), "lf_eq": rng.normal(0, 1, 32).tolist()}
        def build(tag, sd_shift_a, nonsd_shift_a, nonsd_shift_b):
            rows, own = [], []
            for c in ("run3", "dresden"):
                for d in range(10):
                    for i in range(3):
                        rows.append({"key": f"{c}/{d}/{i}", "corpus": c, "group": "real", "device": f"{c}{d}", "status": "ok", "arms": {"R1": feats(0, 0)}})
            for f in SD + NON_SD:
                for i in range(25):
                    sa = sd_shift_a if f in SD else nonsd_shift_a
                    rows.append({"key": f"ai/{f}/{i}", "corpus": "ai", "group": f, "device": None, "status": "ok", "arms": {"A5": feats(sa, 0 if f in SD else nonsd_shift_b)}})
            for i in range(40):
                own.append({"key": f"own/{i}", "corpus": "ai", "group": GEN, "device": None, "status": "ok", "arms": {"A1": feats(1.5, 0), "A5": feats(1.5, 0)}})
            w = lambda n, r: (open(os.path.join(tmp, f"{tag}-{n}.jsonl"), "w").write("\n".join(json.dumps(x) for x in r)), os.path.join(tmp, f"{tag}-{n}.jsonl"))[1]
            return argparse.Namespace(resample_feats=w("rs", rows), own_feats=w("own", own), out=tmp, n_boot=20)
        # linaje: la senal comun esta solo en las familias SD; las no SD tienen otra senal
        r = analyze(build("lin", 1.5, 0, 1.5))
        assert r["veredicto"] == "explicada por linaje SD del entrenamiento", (r["exclusiones"]["X3_sin_linaje_sd"]["media"], r["media_controles"])
        # generico: la misma senal en todas las familias
        r = analyze(build("gen", 1.5, 1.5, 0))
        assert r["veredicto"] == "no explicada por linaje SD", (r["exclusiones"]["X3_sin_linaje_sd"]["media"], r["veredicto"])
        assert r["exclusiones"]["X0_ninguna"]["media"] >= 0.75
        LOG.info("autotest OK (4 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("resample-feats", "own-feats", "out"):
        p.add_argument("--" + k)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("resample_feats", "own_feats", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    os.makedirs(a.out, exist_ok=True)
    analyze(a)


if __name__ == "__main__":
    main()
