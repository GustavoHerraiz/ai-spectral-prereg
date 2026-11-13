#!/usr/bin/env python3
"""H1-R: replicacion EXPLORATORIA con carpetas ai_* de procedencia no verificada. Ver PREREGISTRO_H1R.md.

Extrae A1 y A5 (h1-resample.process) de cada carpeta ai_* (grupo = nombre de carpeta) y reutiliza el analisis (a)/(b)/(c) de h1-chain.py
con los R5/R1 ya extraidos en h1-chain-run1. No modifica nada en las carpetas de entrada. Rechaza (y cuenta) imagenes que no sean 512x512.
Uso:
  python3 h1r-explore.py --datasets-root DIR --resample-feats F --chain-feats F --out OUT [--max-per-group 100] [--workers 2]
  python3 h1r-explore.py --self-test
"""
import argparse, json, logging, os, sys, tempfile
from multiprocessing import Pool
from PIL import Image
import importlib.util

LOG = logging.getLogger("h1r-explore")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


CH = load_module("h1chain", os.path.join(HERE, "h1-chain.py"))
RS = CH.RS
EXCLUDE_SUFFIX = ("_raw",)


def discover(root, max_per_group):
    """Carpetas ai_* con PNG; devuelve items y un resumen de tamanos por carpeta. Orden determinista."""
    items, summary = [], {}
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not (d.startswith("ai_") and os.path.isdir(full)) or d.endswith(EXCLUDE_SUFFIX):
            continue
        files = sorted(f for f in os.listdir(full) if f.lower().endswith(".png"))[:max_per_group]
        ok, bad = [], 0
        for f in files:
            with Image.open(os.path.join(full, f)) as im:
                if im.size == (512, 512):
                    ok.append(f)
                else:
                    bad += 1
        summary[d] = {"png": len(files), "512x512": len(ok), "otro_tamano": bad}
        items += [(os.path.join(full, f), {"key": f"{d}/{f}", "corpus": "ai", "group": d, "device": None}) for f in ok]
    return items, summary


def extract(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h1r-explore-feats.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8") if l.strip()}
    items, summary = discover(a.datasets_root, a.max_per_group)
    for d, s in summary.items():
        LOG.info("carpeta %s: %s", d, s)
    items = [it for it in items if it[1]["key"] not in done]
    LOG.info("%d imagenes por hacer (%d ya hechas)", len(items), len(done))
    with open(path, "a", encoding="utf-8") as out, Pool(a.workers) as pool:
        for n, rec in enumerate(pool.imap_unordered(RS.process, items, chunksize=2), 1):
            out.write(json.dumps(rec) + "\n"); out.flush()
            if n % 50 == 0:
                LOG.info("%d/%d", n, len(items))
    with open(os.path.join(a.out, "h1r-explore-folders.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return path


def self_test():
    import shutil
    import numpy as np
    rng = np.random.default_rng(5)
    tmp = tempfile.mkdtemp()
    try:
        for d, size, n in (("ai_a", 512, 3), ("ai_b_raw", 512, 2), ("ai_c", 600, 2), ("real", 512, 2)):
            os.makedirs(os.path.join(tmp, d))
            for i in range(n):
                Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(os.path.join(tmp, d, f"x_{i}.png"))
        items, summary = discover(tmp, 100)
        assert set(summary) == {"ai_a", "ai_c"}, summary
        assert summary["ai_c"] == {"png": 2, "512x512": 0, "otro_tamano": 2}
        assert len(items) == 3 and all(m["group"] == "ai_a" for _, m in items)
        assert len(discover(tmp, 2)[0]) == 2
        a = argparse.Namespace(datasets_root=tmp, out=os.path.join(tmp, "o"), max_per_group=100, workers=1)
        p = extract(a)
        recs = [json.loads(l) for l in open(p)]
        assert len(recs) == 3 and all(r["status"] == "ok" and {"A1", "A5"} <= set(r["arms"]) for r in recs)
        extract(a)
        assert len([1 for _ in open(p)]) == 3, "debe reanudar sin duplicar"
        LOG.info("autotest OK (6 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("datasets-root", "resample-feats", "chain-feats", "out"):
        p.add_argument("--" + k)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--max-per-group", type=int, default=100)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("datasets_root", "resample_feats", "chain_feats", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    a.own_feats = extract(a)
    CH.analyze(a, a.chain_feats)


if __name__ == "__main__":
    main()
