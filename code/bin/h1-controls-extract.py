#!/usr/bin/env python3
"""Extraccion unica de los controles de formato/geometria de H1-L2 (solo extrae; no evalua).

Misma muestra que h1-extract.py (905 imagenes de run3, sin Dresden). Para cada imagen calcula, bajo varios brazos de
igualacion, `res_spec` (64 bins) y `lf_eq` (32 bins) con exactamente las mismas funciones que h1-extract.py:
  mod8      recorte 512 con origen multiplo de 8 (alinea la rejilla JPEG de 8x8 en reales e IA)
  q75, q92  mod8 + recompresion JPEG 4:2:0 de calidad 75 / 92 a TODAS las imagenes (reales e IA)
  down      reales reducidos (LANCZOS) a lado menor 1024 antes del recorte alineado; IA sin tocar (iguala la escala)
  down_q92  down + JPEG Q92 a todas
Tambien guarda el resto de origen del recorte central original (mod 8). Reanudable; manifest con SHA-256; no toca nada fuera de --out.
Uso:
  python3 h1-controls-extract.py --run-dir RUN3 --real-dir DIR --synth-root DIR --out OUT [--workers 2]
  python3 h1-controls-extract.py --self-test
"""
import argparse, io, json, logging, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image

LOG = logging.getLogger("h1-controls-extract")
HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


EX = load_module("h1extract", os.path.join(HERE, "h1-extract.py"))
PROXY, CORE = EX.PROXY, EX.CORE
DOWN_SHORT = 1024
ARMS = {"mod8": (False, None), "q75": (False, 75), "q92": (False, 92), "down": (True, None), "down_q92": (True, 92)}


def aligned_crop(img):
    w, h = img.size
    left, top = ((w - 512) // 2) // 8 * 8, ((h - 512) // 2) // 8 * 8
    return img.crop((left, top, left + 512, top + 512))


def jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buf.seek(0)
    out = Image.open(buf); out.load()
    return out.convert("RGB"), len(buf.getvalue())


def features(img):
    arr = np.asarray(img, dtype=np.uint8)
    w_gray = CORE.residual(arr.astype(np.float64) / 255.0) @ EX.GRAY
    power = np.abs(np.fft.fft2((w_gray - w_gray.mean()) * EX.HANN)) ** 2
    sums = np.bincount(EX.SPEC_IDX[EX.SPEC_VALID], weights=power[EX.SPEC_VALID], minlength=EX.N_SPEC)
    counts = np.bincount(EX.SPEC_IDX[EX.SPEC_VALID], minlength=EX.N_SPEC)
    res_spec = np.log10(sums / np.maximum(counts, 1) + 1e-18).round(5).tolist()
    eq, _ = PROXY.equalize(Image.fromarray(arr))
    return PROXY.spectral_features(eq).round(5).tolist(), res_spec


def process(item):
    path, meta = item
    try:
        with Image.open(path) as src:
            fmt, nbytes = src.format, os.path.getsize(path)
            img = src.convert("RGB")
        w, h = img.size
        if min(w, h) < 512:
            raise ValueError("menor que 512")
        rec = {**meta, "status": "ok", "off8": [((w - 512) // 2) % 8, ((h - 512) // 2) % 8], "width": w, "height": h, "format": fmt, "bytes": nbytes, "arms": {}}
        for name, (down, q) in ARMS.items():
            src_img = img
            if down and meta["label"] == "real" and min(w, h) > DOWN_SHORT:
                s = DOWN_SHORT / min(w, h)
                src_img = img.resize((max(512, round(w * s)), max(512, round(h * s))), Image.LANCZOS)
            crop = aligned_crop(src_img)
            if q is None:
                info = {"bytes": nbytes, "width": w, "height": h, "format": fmt}
            else:
                crop, nb = jpeg_roundtrip(crop, q)
                info = {"bytes": nb, "width": 512, "height": 512, "format": "JPEG"}
            lf, sp = features(crop)
            rec["arms"][name] = {**info, "lf_eq": lf, "res_spec": sp}
        return rec
    except Exception as exc:
        return {**meta, "status": "error", "error": str(exc)[:200]}


def run(a):
    os.makedirs(a.out, exist_ok=True)
    items = EX.build_items(argparse.Namespace(run_dir=a.run_dir, real_dir=a.real_dir, synth_root=a.synth_root, dresden_glob=None))
    out_path = os.path.join(a.out, "h1-controls.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    manifest = {"script_sha256": EX.sha256_file(os.path.abspath(__file__)), "extract_sha256": EX.sha256_file(os.path.join(HERE, "h1-extract.py")),
                "proxy_sha256": EX.sha256_file(os.path.join(HERE, "l1-proxy-logo.py")), "sample_sha256": EX.sha256_file(os.path.join(a.run_dir, "sample.json")),
                "n_items": len(items), "arms": {k: list(v) for k, v in ARMS.items()}, "down_short": DOWN_SHORT}
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            if json.load(f) != manifest:
                LOG.error("manifest distinto del existente: no se mezclan ejecuciones (borra --out o corrige la entrada)")
                sys.exit(2)
    else:
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
    todo = [it for it in items if it[1]["key"] not in done]
    LOG.info("%d imagenes, %d ya hechas, %d por hacer", len(items), len(done), len(todo))
    n_err = 0
    with open(out_path, "a", encoding="utf-8") as out, Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap(process, todo, chunksize=2), 1):
            out.write(json.dumps(rec) + "\n"); out.flush()
            n_err += rec["status"] != "ok"
            if i % 50 == 0:
                LOG.info("%d/%d (errores %d)", i, len(todo), n_err)
    LOG.info("fin: %d nuevas, errores %d", len(todo), n_err)


def self_test():
    import shutil
    rng = np.random.default_rng(2)
    tmp = tempfile.mkdtemp()
    try:
        yy, xx = np.mgrid[0:1500, 0:1800]
        base = 0.5 + 0.2 * np.sin(xx / 90.0) * np.cos(yy / 70.0)
        big = np.clip(base[..., None] + rng.normal(0, 0.02, (1500, 1800, 3)), 0, 1)
        real_dir = os.path.join(tmp, "reals", "devX"); os.makedirs(real_dir)
        synth = os.path.join(tmp, "synth", "fam"); os.makedirs(synth)
        Image.fromarray((big * 255).astype(np.uint8)).save(os.path.join(real_dir, "a.jpg"), quality=90)
        Image.fromarray((big[:1100, :1100] * 255).astype(np.uint8)).save(os.path.join(synth, "b.png"))
        run_dir = os.path.join(tmp, "run"); os.makedirs(run_dir)
        with open(os.path.join(run_dir, "sample.json"), "w") as f:
            json.dump([{"key": "real/devX/a.jpg", "group": "real", "label": "real", "rel": "devX/a.jpg"},
                       {"key": "fam/fam/b.png", "group": "fam", "label": "fake", "rel": "fam/b.png"}], f)
        a = argparse.Namespace(run_dir=run_dir, real_dir=os.path.join(tmp, "reals"), synth_root=os.path.join(tmp, "synth"), out=os.path.join(tmp, "out"), workers=1)
        run(a)
        with open(os.path.join(a.out, "h1-controls.jsonl")) as f:
            recs = [json.loads(l) for l in f]
        assert len(recs) == 2 and all(r["status"] == "ok" for r in recs), recs
        for r in recs:
            assert set(r["arms"]) == set(ARMS)
            for arm in r["arms"].values():
                assert len(arm["lf_eq"]) == 32 and len(arm["res_spec"]) == 64
            assert r["arms"]["q75"]["format"] == "JPEG" and r["arms"]["q75"]["bytes"] < r["arms"]["q92"]["bytes"]
        real = recs[0] if recs[0]["label"] == "real" else recs[1]
        assert real["off8"] == [((1800 - 512) // 2) % 8, ((1500 - 512) // 2) % 8] and real["arms"]["down"]["width"] == 1800
        assert aligned_crop(Image.new("RGB", (1801, 1503))).size == (512, 512)
        n_before = len(recs)
        run(a)
        with open(os.path.join(a.out, "h1-controls.jsonl")) as f:
            assert len(f.readlines()) == n_before
        with open(os.path.join(run_dir, "sample.json"), "w") as f:
            json.dump([], f)
        try:
            run(a); raise AssertionError("debia abortar")
        except SystemExit as e:
            assert e.code == 2
        LOG.info("autotest OK (6 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--run-dir"); p.add_argument("--real-dir"); p.add_argument("--synth-root"); p.add_argument("--out"); p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("run_dir", "real_dir", "synth_root", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    run(a)


if __name__ == "__main__":
    main()
