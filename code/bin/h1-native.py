#!/usr/bin/env python3
"""H1-L2: generalizacion entre corpus a resolucion NATIVA (sin reescalar). Ver PREREGISTRO_H1.md seccion 13.

Corpus de reales nativos: run3 (205, de h1-controls-run1) y dresden (<=10/camara, regla y semilla de H1; se extrae aqui con los
mismos brazos que h1-controls-extract.py). IA: 700 Synthbuster de h1-controls-run1. Recorte 512 alineado a 8 sin reescalar;
brazos mod8 (referencia), q75 y q92 (JPEG 4:2:0 a TODAS las imagenes: iguala formato). Pares: run3->dresden y dresden->run3.
Uso:
  python3 h1-native.py --dresden-glob 'GLOB' --controls-dir h1-controls-run1 --out OUT [--workers 2]
  python3 h1-native.py --self-test
"""
import argparse, glob, json, logging, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image

LOG = logging.getLogger("h1-native")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


LOCO = load_module("h1loco", os.path.join(HERE, "h1-loco.py"))
CTL, ANA, EX = LOCO.CTL, LOCO.ANA, LOCO.EX
FEATS, ARMS = LOCO.FEATS, ("mod8", "q75", "q92")
PAIRS = (("run3", "dresden"), ("dresden", "run3"))


def dresden_items(glob_):
    items, rng = [], np.random.default_rng(ANA.SEED)
    for d in sorted(glob.glob(glob_)):
        files = sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(LOCO.EXT))
        pick = sorted(rng.choice(len(files), min(EX.DRESDEN_MAX, len(files)), replace=False)) if files else []
        device = os.path.basename(os.path.dirname(d))
        for i in pick:
            items.append((files[i], {"key": f"dresden/{device}/{os.path.basename(files[i])}", "group": "real-dresden", "label": "real", "device": device}))
    return items


def extract(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h1-native-dresden.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8") if l.strip()}
    items = [it for it in dresden_items(a.dresden_glob) if it[1]["key"] not in done]
    LOG.info("dresden: %d por hacer (%d ya hechas)", len(items), len(done))
    with open(path, "a", encoding="utf-8") as out, Pool(a.workers) as pool:
        for rec in pool.imap_unordered(CTL.process, items, chunksize=2):
            out.write(json.dumps(rec) + "\n"); out.flush()
    return path


def analyze(a, dres_path):
    ctl = LOCO.read_jsonl(os.path.join(a.controls_dir, "h1-controls.jsonl"))
    dres = LOCO.read_jsonl(dres_path)
    ai = [r for r in ctl if r["label"] == "fake"]
    r3 = [r for r in ctl if r["label"] == "real"]
    reals = [("run3", r) for r in r3] + [("dresden", r) for r in dres]
    corp = np.array([c for c, _ in reals]); dev = np.array([f'{c}:{r["device"]}' for c, r in reals])
    fam_ai = np.array([r["group"] for r in ai])
    LOG.info("reales: run3 %d, dresden %d; IA %d", len(r3), len(dres), len(ai))
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for feat in FEATS:
            X_ai = np.array([r["arms"][arm][feat] for r in ai]); X_re = np.array([r["arms"][arm][feat] for _, r in reals])
            res = {"cruzado": {f"{a_}->{b}": LOCO.cross_corpus(X_ai, fam_ai, X_re, corp, dev, [a_], b, a.n_boot) for a_, b in PAIRS},
                   "en_dominio": {c: LOCO.in_domain(X_ai, fam_ai, X_re, corp, dev, c, a.n_boot) for c in ("run3", "dresden")}}
            res["veredicto"], res["media_cruzado"], res["min_ic_inf"] = LOCO.verdict(res)
            out[arm][feat] = res
            for grp in ("cruzado", "en_dominio"):
                for k, v in res[grp].items():
                    LOG.info("| %-5s | %-8s | %-10s %-16s | %.3f | [%.3f, %.3f] |", arm, feat, grp, k, v["auc"], *v["ic95"])
            LOG.info("RESUMEN %s %s: media cruzado %.3f, min IC inf %.3f -> %s", arm, feat, res["media_cruzado"], res["min_ic_inf"], res["veredicto"])
    for feat in FEATS:
        LOG.info("VEREDICTO H1-L2-NATIVO %s (brazo q92, formato igualado): %s", feat, out["q92"][feat]["veredicto"])
    with open(os.path.join(a.out, "h1-native.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h1-native.json"))
    return out


def self_test():
    import shutil
    rng = np.random.default_rng(5)
    tmp = tempfile.mkdtemp()
    try:
        yy, xx = np.mgrid[0:700, 0:800]
        def make(ai_, size=(800, 700)):
            tilt = 0.3 * np.sin(xx / rng.uniform(40, 200) + rng.uniform(0, 6)) * np.cos(yy / rng.uniform(40, 200) + rng.uniform(0, 6))
            arr = 0.5 + tilt[..., None] + rng.normal(0, 0.03, (700, 800, 3))
            if ai_:
                arr += 0.05 * np.sin(xx * 2.4)[..., None]
            return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).crop((0, 0) + size)
        dres, cdir, out = (os.path.join(tmp, n) for n in ("d", "ctl", "out"))
        os.makedirs(cdir)
        for c in range(6):
            os.makedirs(os.path.join(dres, f"cam{c}", "nat"))
            for i in range(4): make(False).save(os.path.join(dres, f"cam{c}", "nat", f"d{i}.jpg"), quality=90)
        ctl = []
        def rec(im, label, group, device):
            arms = {}
            for name in ARMS:
                q = CTL.ARMS[name][1]; crop = CTL.aligned_crop(im)
                if q: crop, _ = CTL.jpeg_roundtrip(crop, q)
                lf, sp = CTL.features(crop); arms[name] = {"lf_eq": lf, "res_spec": sp}
            return {"key": f"{group}{rng.integers(1e9)}", "label": label, "group": group, "device": device, "status": "ok", "arms": arms}
        for d in range(8):
            for i in range(3): ctl.append(rec(make(False), "real", "real", f"dev{d}"))
        for fam in "abc":
            for i in range(10): ctl.append(rec(make(True), "fake", fam, None))
        with open(os.path.join(cdir, "h1-controls.jsonl"), "w") as f: f.write("\n".join(json.dumps(r) for r in ctl))
        a = argparse.Namespace(dresden_glob=os.path.join(dres, "*", "nat"), controls_dir=cdir, out=out, workers=1, n_boot=20)
        p = extract(a)
        assert len(LOCO.read_jsonl(p)) == 24 and extract(a) == p and len(LOCO.read_jsonl(p)) == 24, "extraccion/reanudacion"
        res = analyze(a, p)
        assert res["q92"]["res_spec"]["veredicto"] == "aguanta", res["q92"]["res_spec"]["media_cruzado"]
        rows = [json.loads(l) for l in open(p)]
        real_ctl = [r for r in ctl if r["label"] == "real"]
        for i, r in enumerate(r for r in ctl if r["label"] == "fake"):
            r["arms"] = real_ctl[i % len(real_ctl)]["arms"]
        with open(os.path.join(cdir, "h1-controls.jsonl"), "w") as f: f.write("\n".join(json.dumps(r) for r in ctl))
        null = analyze(a, p)["q92"]["res_spec"]
        assert null["veredicto"] != "aguanta" and null["media_cruzado"] < 0.68, null["media_cruzado"]
        LOG.info("autotest OK (4 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("dresden-glob", "controls-dir", "out"):
        p.add_argument("--" + k)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("dresden_glob", "controls_dir", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    analyze(a, extract(a))


if __name__ == "__main__":
    main()
