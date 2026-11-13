#!/usr/bin/env python3
"""H1-L2: curva dosis-respuesta de la escala del contenido. Ver PREREGISTRO_H1.md seccion 15.

Reales nativos (run3 + dresden) reducidos por un factor f (LANCZOS) antes del recorte 512 alineado a 8; extremo "P": reducidos a
lado menor 512. JPEG q92 4:2:0 a TODAS las imagenes (IA de h1-controls-run1, brazo q92, sin tocar). Muestra comun: reales con
lado menor >= 512 * max(f) (la misma en todos los factores). AUC cruzado run3<->dresden (LOFO por familia IA) en cada factor.
Uso:
  python3 h1-scale.py --run-dir RUN3 --real-dir DIR --dresden-glob 'GLOB' --controls-dir h1-controls-run1 --out OUT [--factors 1,1.5,2,2.5,3]
  python3 h1-scale.py --self-test
"""
import argparse, glob, json, logging, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image

LOG = logging.getLogger("h1-scale")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


NAT = load_module("h1native", os.path.join(HERE, "h1-native.py"))
LOCO, CTL, ANA = NAT.LOCO, NAT.CTL, NAT.ANA
FEATS, PAIRS, QUALITY = LOCO.FEATS, NAT.PAIRS, 92
DOWN_MIN_DROP, MONO_TOL, NO_SCALE_MIN = 0.20, 0.03, 0.75  # criterios de la seccion 15
FACTORS = (1.0, 1.5, 2.0, 2.5, 3.0)
KEEP = {}


def arm_name(f):
    return "P" if f == "P" else f"x{f:g}"


def scaled(img, f):
    w, h = img.size
    if f == "P":
        s = 512 / min(w, h)
    elif f == 1.0:
        return img
    else:
        s = 1.0 / f
    return img.resize((max(512, round(w * s)), max(512, round(h * s))), Image.LANCZOS)


def process(item):
    path, meta, factors = item
    try:
        with Image.open(path) as src:
            img = src.convert("RGB")
        if min(img.size) < 512 * max(factors):
            return {**meta, "status": "skip", "short": min(img.size)}
        arms = {}
        for f in list(factors) + ["P"]:
            crop, _ = CTL.jpeg_roundtrip(CTL.aligned_crop(scaled(img, f)), QUALITY)
            lf, sp = CTL.features(crop)
            arms[arm_name(f)] = {"lf_eq": lf, "res_spec": sp}
        return {**meta, "status": "ok", "short": min(img.size), "arms": arms}
    except Exception as exc:
        return {**meta, "status": "error", "error": str(exc)[:200]}


def build_items(a, factors):
    items = []
    if a.run_dir:
        with open(os.path.join(a.run_dir, "sample.json"), encoding="utf-8") as f:
            for s in json.load(f):
                if s["label"] == "real":
                    items.append((os.path.join(a.real_dir, s["rel"]), {"key": "run3/" + s["rel"], "corpus": "run3", "device": s["rel"].split(os.sep)[0]}, factors))
    for p, m in NAT.dresden_items(a.dresden_glob):
        items.append((p, {"key": m["key"], "corpus": "dresden", "device": m["device"]}, factors))
    return items


def extract(a, factors):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h1-scale-feats.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8") if l.strip()}
    items = [it for it in build_items(a, factors) if it[1]["key"] not in done]
    LOG.info("%d reales por hacer (%d ya hechos)", len(items), len(done))
    with open(path, "a", encoding="utf-8") as out, Pool(a.workers) as pool:
        for rec in pool.imap_unordered(process, items, chunksize=2):
            out.write(json.dumps(rec) + "\n"); out.flush()
    return path


def verdict(curve):
    """curve: AUC medio cruzado por factor, en orden creciente de reduccion (el ultimo es P)."""
    first, last = curve[0], curve[-1]
    mono = all(curve[i + 1] <= curve[i] + MONO_TOL for i in range(len(curve) - 1))
    if first - last >= DOWN_MIN_DROP and mono:
        return "dependiente de escala"
    if last >= NO_SCALE_MIN:
        return "no explicada por escala"
    return "indeterminada"


def analyze(a, path, factors):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    LOG.info("reales usables: %s; excluidos por lado menor < %d: %s; errores %d", {c: sum(r["corpus"] == c for r in ok) for c in ("run3", "dresden")}, 512 * max(factors),
             {c: sum(r["corpus"] == c and r["status"] == "skip" for r in rows) for c in ("run3", "dresden")}, sum(r["status"] == "error" for r in rows))
    ctl = [r for r in LOCO.read_jsonl(os.path.join(a.controls_dir, "h1-controls.jsonl")) if r["label"] == "fake"]
    fam_ai = np.array([r["group"] for r in ctl])
    corp = np.array([r["corpus"] for r in ok]); dev = np.array([f'{r["corpus"]}:{r["device"]}' for r in ok])
    out = {"n_reales": {c: int((corp == c).sum()) for c in ("run3", "dresden")}, "factores": [arm_name(f) for f in list(factors) + ["P"]]}
    LOG.info("| rasgo | brazo | run3->dresden | dresden->run3 | media |")
    for feat in FEATS:
        X_ai = np.array([r["arms"]["q92"][feat] for r in ctl]); curve, res = [], {}
        for f in list(factors) + ["P"]:
            X_re = np.array([r["arms"][arm_name(f)][feat] for r in ok])
            pr = {f"{x}->{y}": LOCO.cross_corpus(X_ai, fam_ai, X_re, corp, dev, [x], y, a.n_boot) for x, y in PAIRS}
            m = float(np.mean([v["auc"] for v in pr.values()])); curve.append(m); res[arm_name(f)] = {"pares": pr, "media": m}
            LOG.info("| %-8s | %-5s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f |", feat, arm_name(f), *[x for v in pr.values() for x in (v["auc"], *v["ic95"])], m)
        out[feat] = {"curva": res, "veredicto": verdict(curve)}
        LOG.info("VEREDICTO H1-L2-ESCALA %s: %s (curva %s)", feat, out[feat]["veredicto"], [round(c, 3) for c in curve])
    with open(os.path.join(a.out, "h1-scale.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h1-scale.json"))
    return out


def self_test():
    import shutil
    rng = np.random.default_rng(9)
    tmp = tempfile.mkdtemp()
    factors = (1.0, 2.0, 2.7)
    try:
        W, H = 1600, 1400
        yy, xx = np.mgrid[0:H, 0:W]
        def make(real, pattern):
            tilt = 0.3 * np.sin(xx / rng.uniform(80, 400) + rng.uniform(0, 6)) * np.cos(yy / rng.uniform(80, 400) + rng.uniform(0, 6))
            world = Image.fromarray((np.clip(0.5 + tilt[..., None] + rng.normal(0, 0.07, (H, W, 3)), 0, 1) * 255).astype(np.uint8))
            if real:
                return world
            # IA = el mismo mundo ya a 512 (misma cadena de reescalado); con `pattern` ademas una huella propia
            small = scaled(world, "P").crop((0, 0, 512, 512))
            if pattern:
                arr = np.asarray(small, dtype=np.float64) / 255 + 0.06 * np.sin(np.arange(512) * 2.4)[None, :, None]
                small = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            return small
        def build(pattern, tag):
            root = os.path.join(tmp, tag); real_dir, run_dir, dres, cdir, out = (os.path.join(root, n) for n in ("r", "run", "d", "ctl", "out"))
            for d in (run_dir, cdir): os.makedirs(d)
            sample, ctl = [], []
            for d in range(6):
                os.makedirs(os.path.join(real_dir, f"dev{d}"))
                for i in range(2):
                    make(True, pattern).save(os.path.join(real_dir, f"dev{d}", f"x{i}.jpg"), quality=92)
                    sample.append({"key": f"r{d}{i}", "group": "real", "label": "real", "rel": f"dev{d}/x{i}.jpg"})
            for c in range(4):
                os.makedirs(os.path.join(dres, f"cam{c}", "nat"))
                for i in range(3): make(True, pattern).save(os.path.join(dres, f"cam{c}", "nat", f"d{i}.jpg"), quality=90)
            for fam in "abc":
                for i in range(10):
                    im = make(False, pattern); crop, _ = CTL.jpeg_roundtrip(im, QUALITY); lf, sp = CTL.features(crop)
                    ctl.append({"key": f"{fam}{i}", "label": "fake", "group": fam, "device": None, "status": "ok", "arms": {"q92": {"lf_eq": lf, "res_spec": sp}}})
            with open(os.path.join(run_dir, "sample.json"), "w") as f: json.dump(sample, f)
            with open(os.path.join(cdir, "h1-controls.jsonl"), "w") as f: f.write("\n".join(json.dumps(r) for r in ctl))
            return argparse.Namespace(run_dir=run_dir, real_dir=real_dir, dresden_glob=os.path.join(dres, "*", "nat"), controls_dir=cdir, out=out, workers=1, n_boot=15)
        a = build(False, "esc")
        p = extract(a, factors)
        assert sum(json.loads(l)["status"] == "ok" for l in open(p)) == 24, "extraccion"
        assert extract(a, factors) == p and len(open(p).readlines()) == 24, "reanudacion"
        r = analyze(a, p, factors)["res_spec"]
        assert r["veredicto"] == "dependiente de escala", (r["veredicto"], [v["media"] for v in r["curva"].values()])
        b = build(True, "pat")
        rp = analyze(b, extract(b, factors), factors)["res_spec"]
        assert rp["veredicto"] == "no explicada por escala", (rp["veredicto"], [v["media"] for v in rp["curva"].values()])
        assert verdict([0.9, 0.8, 0.5]) == "dependiente de escala" and verdict([0.9, 0.5, 0.7]) == "indeterminada"
        LOG.info("autotest OK (5 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("run-dir", "real-dir", "dresden-glob", "controls-dir", "out"):
        p.add_argument("--" + k)
    p.add_argument("--factors", default="1,1.5,2,2.5,3"); p.add_argument("--workers", type=int, default=2); p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("run_dir", "real_dir", "dresden_glob", "controls_dir", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    factors = tuple(float(x) for x in a.factors.split(","))
    analyze(a, extract(a, factors), factors)


if __name__ == "__main__":
    main()
