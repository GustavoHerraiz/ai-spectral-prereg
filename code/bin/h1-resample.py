#!/usr/bin/env python3
"""H1-L2: control de remuestreo y de doble compresion para res_spec/lf_eq. Ver PREREGISTRO_H1.md seccion 17.

Reales (run3 + dresden): reducidos a lado menor 512 (LANCZOS), recorte alineado a 8; R0 sin JPEG, R1 con JPEG q92.
IA Synthbuster (700, de sample.json): recorte 512 alineado a 8 y cadenas que imitan lo que sufren los reales:
  A0 nada | A1 q92 | A2 j90+q92 (doble compresion) | A3 lp (LANCZOS x1.5 arriba y abajo: paso bajo/remuestreo)
  A4 lp+q92 | A5 lp+j90+q92 (todo igualado) | A6 lp+j90 (todo igualado sin JPEG final)
Comparaciones (IA, real): C1 A0/R0 | C2 A1/R1 | C3 A2/R1 | C4 A4/R1 | C5 A5/R1 (decisiva) | C6 A6/R0.
AUC cruzado run3<->dresden (LOFO por familia IA), como h1-native.py.
Uso:
  python3 h1-resample.py --run-dir RUN3 --real-dir DIR --synth-root DIR --dresden-glob 'GLOB' --out OUT [--workers 2]
  python3 h1-resample.py --self-test
"""
import argparse, json, logging, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image

LOG = logging.getLogger("h1-resample")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


NAT = load_module("h1native", os.path.join(HERE, "h1-native.py"))
LOCO, CTL = NAT.LOCO, NAT.CTL
FEATS, PAIRS = LOCO.FEATS, NAT.PAIRS
COMPS = {"C1": ("A0", "R0"), "C2": ("A1", "R1"), "C3": ("A2", "R1"), "C4": ("A4", "R1"), "C5": ("A5", "R1"), "C6": ("A6", "R0")}
DECISIVE = "C5"
LP_UP = 1.5


def jpeg(img, q):
    return CTL.jpeg_roundtrip(img, q)[0]


def lp(img):
    w, h = img.size
    return img.resize((round(w * LP_UP), round(h * LP_UP)), Image.LANCZOS).resize((w, h), Image.LANCZOS)


def feats(img):
    lf, sp = CTL.features(img)
    return {"lf_eq": lf, "res_spec": sp}


def process(item):
    path, meta = item
    try:
        with Image.open(path) as src:
            img = src.convert("RGB")
        if min(img.size) < 512:
            raise ValueError("menor que 512")
        if meta["corpus"] == "ai":
            c = CTL.aligned_crop(img); l = lp(c)
            chains = {"A0": c, "A1": jpeg(c, 92), "A2": jpeg(jpeg(c, 90), 92), "A3": l, "A4": jpeg(l, 92), "A5": jpeg(jpeg(l, 90), 92), "A6": jpeg(l, 90)}
        else:
            w, h = img.size; s = 512 / min(w, h)
            c = CTL.aligned_crop(img.resize((max(512, round(w * s)), max(512, round(h * s))), Image.LANCZOS))
            chains = {"R0": c, "R1": jpeg(c, 92)}
        return {**meta, "status": "ok", "arms": {k: feats(v) for k, v in chains.items()}}
    except Exception as exc:
        return {**meta, "status": "error", "error": str(exc)[:200]}


def build_items(a):
    items = []
    with open(os.path.join(a.run_dir, "sample.json"), encoding="utf-8") as f:
        for s in json.load(f):
            if s["label"] == "real":
                items.append((os.path.join(a.real_dir, s["rel"]), {"key": "run3/" + s["rel"], "corpus": "run3", "group": "real", "device": s["rel"].split(os.sep)[0]}))
            else:
                items.append((os.path.join(a.synth_root, s["rel"]), {"key": "ai/" + s["rel"], "corpus": "ai", "group": s["group"], "device": None}))
    for p, m in NAT.dresden_items(a.dresden_glob):
        items.append((p, {"key": m["key"], "corpus": "dresden", "group": "real", "device": m["device"]}))
    assert len({m["key"] for _, m in items}) == len(items), "claves duplicadas"
    return items


def extract(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h1-resample-feats.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8") if l.strip()}
    items = [it for it in build_items(a) if it[1]["key"] not in done]
    LOG.info("%d imagenes por hacer (%d ya hechas)", len(items), len(done))
    n = 0
    with open(path, "a", encoding="utf-8") as out, Pool(a.workers) as pool:
        for rec in pool.imap_unordered(process, items, chunksize=4):
            out.write(json.dumps(rec) + "\n"); out.flush(); n += 1
            if n % 100 == 0:
                LOG.info("%d/%d", n, len(items))
    return path


def analyze(a, path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    ai = [r for r in ok if r["corpus"] == "ai"]; re_ = [r for r in ok if r["corpus"] != "ai"]
    LOG.info("IA %d, reales %s, errores %d", len(ai), {c: sum(r["corpus"] == c for r in re_) for c in ("run3", "dresden")}, sum(r["status"] == "error" for r in rows))
    fam_ai = np.array([r["group"] for r in ai]); corp = np.array([r["corpus"] for r in re_]); dev = np.array([f'{r["corpus"]}:{r["device"]}' for r in re_])
    out = {}
    LOG.info("| rasgo | comp | IA/real | run3->dresden | dresden->run3 | media | veredicto |")
    for feat in FEATS:
        out[feat] = {}
        for name, (aa, ra) in COMPS.items():
            X_ai = np.array([r["arms"][aa][feat] for r in ai]); X_re = np.array([r["arms"][ra][feat] for r in re_])
            pr = {f"{x}->{y}": LOCO.cross_corpus(X_ai, fam_ai, X_re, corp, dev, [x], y, a.n_boot) for x, y in PAIRS}
            v, mean, lo = LOCO.verdict({"cruzado": pr})
            out[feat][name] = {"ia": aa, "real": ra, "pares": pr, "media": mean, "min_ic_inf": lo, "veredicto_base": v}
            LOG.info("| %-8s | %s | %s/%s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f | %s |", feat, name, aa, ra, *[x for p in pr.values() for x in (p["auc"], *p["ic95"])], mean, v)
        v = out[feat][DECISIVE]["veredicto_base"]
        out[feat]["veredicto"] = {"aguanta": "persiste con procesado igualado", "artefacto de corpus": "explicada por procesado"}.get(v, "indeterminada")
        LOG.info("VEREDICTO H1-L2-REMUESTREO %s (%s): %s (media %.3f, min IC inf %.3f)", feat, DECISIVE, out[feat]["veredicto"], out[feat][DECISIVE]["media"], out[feat][DECISIVE]["min_ic_inf"])
    with open(os.path.join(a.out, "h1-resample.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h1-resample.json"))
    return out


def self_test():
    import shutil
    rng = np.random.default_rng(13)
    tmp = tempfile.mkdtemp()
    try:
        W, H = 1600, 1400
        yy, xx = np.mgrid[0:H, 0:W]
        def world():
            tilt = 0.3 * np.sin(xx / rng.uniform(80, 400) + rng.uniform(0, 6)) * np.cos(yy / rng.uniform(80, 400) + rng.uniform(0, 6))
            return Image.fromarray((np.clip(0.5 + tilt[..., None] + rng.normal(0, 0.07, (H, W, 3)), 0, 1) * 255).astype(np.uint8))
        def build(pattern, tag):
            root = os.path.join(tmp, tag); real_dir, run_dir, synth, dres, out = (os.path.join(root, n) for n in ("r", "run", "s", "d", "out"))
            os.makedirs(run_dir); os.makedirs(os.path.join(synth, "x"))
            sample = []
            for d in range(6):
                os.makedirs(os.path.join(real_dir, f"dev{d}"))
                for i in range(2):
                    world().save(os.path.join(real_dir, f"dev{d}", f"x{i}.jpg"), quality=92)
                    sample.append({"key": f"r{d}{i}", "group": "real", "label": "real", "rel": f"dev{d}/x{i}.jpg"})
            for c in range(4):
                os.makedirs(os.path.join(dres, f"cam{c}", "nat"))
                for i in range(3): world().save(os.path.join(dres, f"cam{c}", "nat", f"d{i}.jpg"), quality=90)
            for fam in "abc":
                os.makedirs(os.path.join(synth, fam))
                for i in range(10):
                    w = world()
                    if not pattern: w = jpeg(w, 92)  # nulo: mismo historial (JPEG nativo + reduccion) que los reales
                    small = w.resize((585, 512), Image.LANCZOS)
                    if pattern:
                        arr = np.asarray(small, dtype=np.float64) / 255 + 0.06 * np.sin(np.arange(585) * 2.4)[None, :, None]
                        small = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                    small.save(os.path.join(synth, fam, f"{i}.png"))
                    sample.append({"key": f"{fam}{i}", "group": fam, "label": "fake", "rel": f"{fam}/{i}.png"})
            with open(os.path.join(run_dir, "sample.json"), "w") as f: json.dump(sample, f)
            return argparse.Namespace(run_dir=run_dir, real_dir=real_dir, synth_root=synth, dresden_glob=os.path.join(dres, "*", "nat"), out=out, workers=1, n_boot=15)
        a = build(False, "nul"); p = extract(a)
        assert sum(json.loads(l)["status"] == "ok" for l in open(p)) == 54, "extraccion"
        assert extract(a) == p and len(open(p).readlines()) == 54, "reanudacion"
        r0 = analyze(a, p)["res_spec"]
        assert r0["C1"]["veredicto_base"] == "artefacto de corpus", (r0["C1"]["veredicto_base"], r0["C1"]["media"])  # IA con la misma cadena que los reales
        b = build(True, "pat"); rp = analyze(b, extract(b))["res_spec"]
        assert rp["veredicto"] == "persiste con procesado igualado", (rp["veredicto"], rp[DECISIVE]["media"])
        LOG.info("autotest OK (4 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("run-dir", "real-dir", "synth-root", "dresden-glob", "out"):
        p.add_argument("--" + k)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("run_dir", "real_dir", "synth_root", "dresden_glob", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    analyze(a, extract(a))


if __name__ == "__main__":
    main()
