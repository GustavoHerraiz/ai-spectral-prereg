#!/usr/bin/env python3
"""H1-L2: control de cadena. Ver PREREGISTRO_H1.md seccion 21.

Pregunta: la separacion de la IA propia (seccion 20, brazo A5/R1) refleja la cadena A5 (paso bajo + q90 + q92) y no la generacion?
Se aplica la MISMA cadena A5 a los reales (R5 = real reducido a 512, paso bajo x1.5, q90, q92). Tres analisis:
  (c) PRIMARIO, cadena igualada en ambos lados: entrena Synthbuster A5 + R5(X); test IA propia A5 vs R5(Y), run3<->dresden.
  (a) diagnostico: modelo A5/R1; AUC IA propia A5 > IA propia A1 (mismas imagenes). Alto = el modelo detecta la cadena.
  (b) diagnostico: modelo A5/R1; AUC R5(Y) > R1(Y) (mismos reales). Alto = el modelo marca la cadena tambien en reales.
Reutiliza h1-resample-feats.jsonl (Synthbuster A5, reales R1) y h1-owngen-feats.jsonl (IA propia A1/A5). Solo extrae R5 de los reales.
Uso:
  python3 h1-chain.py --run-dir RUN3 --real-dir DIR --synth-root DIR --dresden-glob 'GLOB' --resample-feats F --own-feats F --out OUT [--workers 2]
  python3 h1-chain.py --self-test
"""
import argparse, json, logging, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image

LOG = logging.getLogger("h1-chain")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


EV = load_module("h1owngeneval", os.path.join(HERE, "h1-owngen-eval.py"))
RS, LOCO, ANA = EV.RS, EV.LOCO, EV.ANA
FEATS, PAIRS = LOCO.FEATS, EV.PAIRS


def real_r5(path):
    with Image.open(path) as src:
        img = src.convert("RGB")
    if min(img.size) < 512:
        raise ValueError("menor que 512")
    w, h = img.size; s = 512 / min(w, h)
    c = RS.CTL.aligned_crop(img.resize((max(512, round(w * s)), max(512, round(h * s))), Image.LANCZOS))
    return RS.jpeg(RS.jpeg(RS.lp(c), 90), 92)


def process_real(item):
    path, meta = item
    try:
        return {**meta, "status": "ok", "arms": {"R5": RS.feats(real_r5(path))}}
    except Exception as exc:
        return {**meta, "status": "error", "error": str(exc)[:200]}


def extract(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h1-chain-feats.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8") if l.strip()}
    items = [it for it in RS.build_items(a) if it[1]["corpus"] != "ai" and it[1]["key"] not in done]
    LOG.info("%d reales por hacer (%d ya hechos)", len(items), len(done))
    with open(path, "a", encoding="utf-8") as out, Pool(a.workers) as pool:
        for n, rec in enumerate(pool.imap_unordered(process_real, items, chunksize=2), 1):
            out.write(json.dumps(rec) + "\n"); out.flush()
            if n % 50 == 0:
                LOG.info("%d/%d", n, len(items))
    return path


def read(path):
    return [r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip()) if r["status"] == "ok"]


def boot_paired(s_hi, s_lo, groups, n_boot, rng):
    """AUC s_hi > s_lo con remuestreo por grupo (dispositivo) o por imagen si groups es None; pares alineados."""
    n = len(s_hi); v = []
    if groups is None:
        for _ in range(n_boot):
            i = rng.choice(n, n); v.append(ANA.auc(s_hi[i], s_lo[i]))
    else:
        ug = sorted(set(groups)); idx = {g: np.where(np.asarray(groups) == g)[0] for g in ug}
        for _ in range(n_boot):
            sel = np.concatenate([idx[ug[i]] for i in rng.choice(len(ug), len(ug))]); v.append(ANA.auc(s_hi[sel], s_lo[sel]))
    return LOCO.ci(v)


def fit(X_pos, X_neg):
    return ANA.model().fit(np.vstack([X_pos, X_neg]), np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))])


def summarize(res, labels):
    v, mean, lo = LOCO.verdict({"cruzado": res})
    return {"pares": res, "media": mean, "min_ic_inf": lo, "veredicto": labels.get(v, "indeterminada")}


def analyze(a, chain_path):
    rs = read(a.resample_feats); chain = {r["key"]: r for r in read(chain_path)}
    own = read(a.own_feats)
    syn = [r for r in rs if r["corpus"] == "ai"]
    reals = [r for r in rs if r["corpus"] != "ai" and r["key"] in chain]
    corp = np.array([r["corpus"] for r in reals]); dev = np.array([r["device"] for r in reals])
    gens = sorted({r["group"] for r in own})
    LOG.info("Synthbuster %d; reales con R5 %s; IA propia %s", len(syn), {c: int((corp == c).sum()) for c in ("run3", "dresden")}, {g: sum(r["group"] == g for r in own) for g in gens})
    out = {}
    primary = {"aguanta": "persiste con cadena igualada", "artefacto de corpus": "artefacto de cadena"}
    detect = {"aguanta": "cadena detectada", "artefacto de corpus": "cadena no detectada"}
    LOG.info("| rasgo | analisis | generador | run3->dresden | dresden->run3 | media | veredicto |")
    for feat in FEATS:
        X_syn = np.array([r["arms"]["A5"][feat] for r in syn])
        X_r1 = np.array([r["arms"]["R1"][feat] for r in reals]); X_r5 = np.array([chain[r["key"]]["arms"]["R5"][feat] for r in reals])
        out[feat] = {}
        for g in gens:
            og = [r for r in own if r["group"] == g]
            O5 = np.array([r["arms"]["A5"][feat] for r in og]); O1 = np.array([r["arms"]["A1"][feat] for r in og])
            ra, rb, rc = {}, {}, {}
            for x, y in PAIRS:
                tx, ty = corp == x, corp == y
                rng = lambda: np.random.default_rng(ANA.SEED)
                m = fit(X_syn, X_r1[tx]); s5, s1 = m.decision_function(O5), m.decision_function(O1)
                ra[f"{x}->{y}"] = {"auc": ANA.auc(s5, s1), "ic95": boot_paired(s5, s1, None, a.n_boot, rng()), "n": len(og)}
                r5y, r1y = m.decision_function(X_r5[ty]), m.decision_function(X_r1[ty])
                rb[f"{x}->{y}"] = {"auc": ANA.auc(r5y, r1y), "ic95": boot_paired(r5y, r1y, dev[ty], a.n_boot, rng()), "n": int(ty.sum())}
                mc = fit(X_syn, X_r5[tx]); so, sr = mc.decision_function(O5), mc.decision_function(X_r5[ty])
                rc[f"{x}->{y}"] = {"auc": ANA.auc(so, sr), "ic95": EV.boot_auc(so, sr, dev[ty], a.n_boot, rng()), "n_own": len(og)}
            res = {"c_cadena_igualada": summarize(rc, primary), "a_ia_propia_A5_vs_A1": summarize(ra, detect), "b_reales_R5_vs_R1": summarize(rb, detect)}
            out[feat][g] = res
            for k, r in res.items():
                p = list(r["pares"].values())
                LOG.info("| %-8s | %-22s | %-8s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f | %s |", feat, k, g, p[0]["auc"], *p[0]["ic95"], p[1]["auc"], *p[1]["ic95"], r["media"], r["veredicto"])
        LOG.info("VEREDICTO H1-L2-CADENA %s (primario c, sdturbo): %s", feat, out[feat].get("sdturbo", {}).get("c_cadena_igualada", {}).get("veredicto", "sin sdturbo"))
    with open(os.path.join(a.out, "h1-chain.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h1-chain.json"))
    return out


def self_test():
    import shutil
    rng = np.random.default_rng(31)
    tmp = tempfile.mkdtemp()
    try:
        def feats(mean):
            return {"res_spec": rng.normal(mean, 1, 64).tolist(), "lf_eq": rng.normal(0, 1, 32).tolist()}
        def build(tag, shift_chain, shift_gen):
            # shift_chain: efecto de la cadena A5/R5 en ambos lados; shift_gen: efecto de ser IA en cualquier cadena
            rows, chain, own = [], [], []
            for c in ("run3", "dresden"):
                for d in range(10):
                    for i in range(3):
                        k = f"{c}/{d}/{i}"
                        rows.append({"key": k, "corpus": c, "group": "real", "device": f"{c}{d}", "status": "ok", "arms": {"R1": feats(0.0)}})
                        chain.append({"key": k, "corpus": c, "group": "real", "device": f"{c}{d}", "status": "ok", "arms": {"R5": feats(shift_chain)}})
            for f in range(4):
                for i in range(25):
                    rows.append({"key": f"ai/{f}/{i}", "corpus": "ai", "group": f"fam{f}", "device": None, "status": "ok", "arms": {"A5": feats(shift_chain + shift_gen)}})
            for i in range(40):
                own.append({"key": f"own/{i}", "corpus": "ai", "group": "sdturbo", "device": None, "status": "ok", "arms": {"A5": feats(shift_chain + shift_gen), "A1": feats(shift_gen)}})
            w = lambda n, r: (open(os.path.join(tmp, f"{tag}-{n}.jsonl"), "w").write("\n".join(json.dumps(x) for x in r)), os.path.join(tmp, f"{tag}-{n}.jsonl"))[1]
            return argparse.Namespace(resample_feats=w("rs", rows), own_feats=w("own", own), out=tmp, n_boot=20), w("ch", chain)
        a, cp = build("cad", 0.8, 0.0)
        r = analyze(a, cp)["res_spec"]["sdturbo"]
        assert r["c_cadena_igualada"]["veredicto"] == "artefacto de cadena", r["c_cadena_igualada"]["media"]
        assert r["a_ia_propia_A5_vs_A1"]["veredicto"] == "cadena detectada" and r["b_reales_R5_vs_R1"]["veredicto"] == "cadena detectada"
        a, cp = build("gen", 0.0, 0.8)
        r = analyze(a, cp)["res_spec"]["sdturbo"]
        assert r["c_cadena_igualada"]["veredicto"] == "persiste con cadena igualada", r["c_cadena_igualada"]["media"]
        assert r["b_reales_R5_vs_R1"]["veredicto"] == "cadena no detectada" and r["a_ia_propia_A5_vs_A1"]["veredicto"] == "cadena no detectada"
        rng2 = np.random.default_rng(3)
        img = Image.fromarray(rng2.integers(0, 255, (900, 1200, 3), dtype=np.uint8)); p = os.path.join(tmp, "x.png"); img.save(p)
        out = real_r5(p)
        assert out.size[0] % 8 == 0 and out.size[1] % 8 == 0 and min(out.size) >= 512, out.size
        LOG.info("autotest OK (6 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("run-dir", "real-dir", "synth-root", "dresden-glob", "resample-feats", "own-feats", "out"):
        p.add_argument("--" + k)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("run_dir", "real_dir", "synth_root", "dresden_glob", "resample_feats", "own_feats", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    analyze(a, extract(a))


if __name__ == "__main__":
    main()
