#!/usr/bin/env python3
"""Extraccion unica de datos para H1-PRNU y H1-L2 (solo extrae; NO evalua ni decide nada).

Para cada imagen de la muestra (905 de la sonda v1 run3 + reales Dresden opcionales) guarda UNA vez:
  - prnu: NCC (Pearson de gris del residuo W contra gris de K) contra TODAS las K del registro (orden en kgallery.json).
    Recorte central nativo 512, pixeles almacenados sin EXIF-transpose, mediana 3x3 reflect (identico a v3/bin/registry_core.py).
  - lf_eq / lf_raw: vector L1-proxy de 32 bins en [0.02, 0.35) (con y sin pipeline igualado 256 + JPEG Q90), de l1-proxy-logo.py.
  - res_spec: log10 de la potencia radial media del residuo W (64 bins logaritmicos en [0.004, 0.5] ciclos/pixel, ventana Hann).
  - metadatos: width, height, bytes, format, group, label, device, rel.
Es reanudable (clave unica por imagen) y no modifica nada fuera de --out. No lee ninguna etiqueta para decidir.
Uso:
  python3 h1-extract.py --run-dir RUN3 --real-dir DIR --synth-root DIR --registry-root DIR --out OUT [--dresden-glob 'GLOB'] [--workers 2]
  python3 h1-extract.py --self-test
"""
import argparse, glob, hashlib, importlib.util, json, logging, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

LOG = logging.getLogger("h1-extract")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SEED = 20261111
N_SPEC = 64
SPEC_LO, SPEC_HI = 0.004, 0.5
DRESDEN_MAX = 10
GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


PROXY = load_module("l1proxy", os.path.join(HERE, "l1-proxy-logo.py"))
CORE = load_module("registry_core", os.path.join(ROOT, "v3", "bin", "registry_core.py"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spec_bins():
    f = np.fft.fftfreq(512)
    r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    edges = np.logspace(np.log10(SPEC_LO), np.log10(SPEC_HI), N_SPEC + 1)
    idx = np.digitize(r, edges) - 1
    return idx, (idx >= 0) & (idx < N_SPEC)


SPEC_IDX, SPEC_VALID = spec_bins()
HANN = np.outer(np.hanning(512), np.hanning(512))


def centered_unit(gray):
    v = gray.ravel() - gray.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def load_gallery(registry_root):
    """K en gris, centradas y normalizadas (float32): (n_k, 512*512). Devuelve ids, matriz, descartadas."""
    with open(os.path.join(registry_root, "registry_index.json"), encoding="utf-8") as f:
        entries = json.load(f)["entries"]
    if isinstance(entries, dict):
        entries = list(entries.values())
    ids, rows, skipped, shas = [], [], [], {}
    for entry in sorted(entries, key=lambda e: e["entryId"]):
        path = entry.get("kPath") or os.path.join(registry_root, "entries", entry["entryId"], "k.npz")
        try:
            k = CORE.load_k_from_npz(CORE.Path(path))
            if k.shape != (512, 512, 3):
                raise ValueError(f"forma {k.shape}")
            unit = centered_unit(k @ GRAY)
            if unit is None:
                raise ValueError("K constante")
        except Exception as exc:
            LOG.warning("K descartada %s: %s", entry["entryId"], exc)
            skipped.append({"entryId": entry["entryId"], "motivo": str(exc)})
            continue
        ids.append(entry["entryId"]); rows.append(unit.astype(np.float32)); shas[entry["entryId"]] = entry.get("npzSha256")
    return ids, np.vstack(rows), skipped, shas


GAL = {}


def init_worker(registry_root):
    GAL["ids"], GAL["mat"], _, _ = load_gallery(registry_root)


def extract_one(item):
    path, meta = item
    try:
        rgb = CORE.load_rgb_native512(CORE.Path(path), exif_transpose=False)
        w_gray = CORE.residual(rgb) @ GRAY
        unit = centered_unit(w_gray)
        scores = None if unit is None else (GAL["mat"] @ unit.astype(np.float32)).astype(float).round(6).tolist()
        power = np.abs(np.fft.fft2((w_gray - w_gray.mean()) * HANN)) ** 2
        sums = np.bincount(SPEC_IDX[SPEC_VALID], weights=power[SPEC_VALID], minlength=N_SPEC)
        counts = np.bincount(SPEC_IDX[SPEC_VALID], minlength=N_SPEC)
        res_spec = np.log10(sums / np.maximum(counts, 1) + 1e-18).round(5).tolist()
        feat_eq, feat_raw, info = PROXY.extract(path)
        with Image.open(path) as im:
            fmt, size = im.format, im.size
        return {**meta, "status": "ok", "format": fmt, "width": size[0], "height": size[1], "bytes": os.path.getsize(path),
                "prnu": scores, "w_std": float(w_gray.std()), "res_spec": res_spec,
                "lf_eq": None if feat_eq is None else feat_eq.round(5).tolist(),
                "lf_raw": None if feat_raw is None else feat_raw.round(5).tolist()}
    except Exception as exc:
        return {**meta, "status": "error", "error": str(exc)[:200]}


def build_items(a):
    with open(os.path.join(a.run_dir, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    items = []
    for s in sample:
        if s["label"] == "real":
            path, device = os.path.join(a.real_dir, s["rel"]), s["rel"].split(os.sep)[0]
        else:
            path, device = os.path.join(a.synth_root, s["rel"]), None
        items.append((path, {"key": s["key"], "group": s["group"], "label": s["label"], "rel": s["rel"], "device": device, "corpus": "run3"}))
    if a.dresden_glob:
        rng = np.random.default_rng(SEED)
        for d in sorted(glob.glob(a.dresden_glob)):
            files = sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
            pick = sorted(rng.choice(len(files), min(DRESDEN_MAX, len(files)), replace=False)) if files else []
            device = os.path.basename(os.path.dirname(d))
            for i in pick:
                key = f"dresden/{device}/{os.path.basename(files[i])}"
                items.append((files[i], {"key": key, "group": "real-dresden", "label": "real", "rel": key, "device": device, "corpus": "dresden"}))
    keys = [m["key"] for _, m in items]
    assert len(keys) == len(set(keys)), "claves duplicadas"
    return items


def run(a):
    os.makedirs(a.out, exist_ok=True)
    ids, mat, skipped, shas = load_gallery(a.registry_root)
    LOG.info("galeria: %d K validas, %d descartadas", len(ids), len(skipped))
    items = build_items(a)
    out_path = os.path.join(a.out, "h1-features.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    manifest = {"script_sha256": sha256_file(os.path.abspath(__file__)), "proxy_sha256": sha256_file(os.path.join(HERE, "l1-proxy-logo.py")),
                "registry_core_sha256": sha256_file(os.path.join(ROOT, "v3", "bin", "registry_core.py")),
                "registry_index_sha256": sha256_file(os.path.join(a.registry_root, "registry_index.json")),
                "sample_sha256": sha256_file(os.path.join(a.run_dir, "sample.json")), "n_items": len(items), "k_ids": ids,
                "k_skipped": skipped, "k_npz_sha256": shas, "seed": SEED, "n_spec": N_SPEC, "spec_range": [SPEC_LO, SPEC_HI], "dresden_max": DRESDEN_MAX}
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            old = json.load(f)
        if old != manifest:
            LOG.error("manifest distinto del existente: no se mezclan ejecuciones (borra --out o corrige la entrada)")
            sys.exit(2)
    else:
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
    todo = [it for it in items if it[1]["key"] not in done]
    LOG.info("%d imagenes, %d ya hechas, %d por hacer", len(items), len(done), len(todo))
    n_err = 0
    with open(out_path, "a", encoding="utf-8") as out, Pool(a.workers, initializer=init_worker, initargs=(a.registry_root,)) as pool:
        for i, rec in enumerate(pool.imap(extract_one, todo, chunksize=2), 1):
            out.write(json.dumps(rec) + "\n"); out.flush()
            n_err += rec["status"] != "ok"
            if i % 50 == 0:
                LOG.info("%d/%d (errores %d)", i, len(todo), n_err)
    LOG.info("fin: %d nuevas, errores %d", len(todo), n_err)


def self_test():
    import io, shutil
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp()
    try:
        reg = os.path.join(tmp, "reg"); os.makedirs(reg)
        ks = {}
        entries = []
        for name in ("camA", "camB"):
            k = rng.normal(0, 0.02, (512, 512, 3)); ks[name] = k
            d = os.path.join(reg, "entries", name); os.makedirs(d); np.savez(os.path.join(d, "k.npz"), K=k)
            entries.append({"entryId": name, "kPath": os.path.join(d, "k.npz"), "npzSha256": "x"})
        with open(os.path.join(reg, "registry_index.json"), "w") as f:
            json.dump({"entries": entries}, f)
        ids, mat, skipped, _ = load_gallery(reg)
        assert ids == ["camA", "camB"] and mat.shape == (2, 512 * 512) and not skipped
        # imagen con huella de camA: base suave * (1 + K)
        yy, xx = np.mgrid[0:600, 0:700]
        base = 0.5 + 0.2 * np.sin(xx / 90.0) * np.cos(yy / 70.0)
        img = np.clip(base[..., None] * np.ones(3) , 0, 1)
        crop = np.s_[44:556, 94:606]
        img[crop] = np.clip(img[crop] * (1 + 8 * ks["camA"]), 0, 1)
        own = os.path.join(tmp, "own.png"); Image.fromarray((img * 255).round().astype(np.uint8)).save(own)
        plain = os.path.join(tmp, "plain.png"); Image.fromarray((np.clip(base[..., None] * np.ones(3), 0, 1) * 255).round().astype(np.uint8)).save(plain)
        GAL["ids"], GAL["mat"] = ids, mat
        r_own = extract_one((own, {"key": "own"})); r_plain = extract_one((plain, {"key": "plain"}))
        assert r_own["status"] == "ok" and r_plain["status"] == "ok", (r_own, r_plain)
        assert r_own["prnu"][0] > 0.1 and r_own["prnu"][0] > 5 * abs(r_own["prnu"][1]), r_own["prnu"]
        assert abs(r_plain["prnu"][0]) < 0.05, r_plain["prnu"]
        # equivalencia con el calculo de referencia del repo
        rgb = CORE.load_rgb_native512(CORE.Path(own), exif_transpose=False)
        ref = CORE.pearson_gray_residual_vs_k(rgb, ks["camA"])
        assert abs(ref - r_own["prnu"][0]) < 1e-4, (ref, r_own["prnu"][0])
        assert len(r_own["res_spec"]) == N_SPEC and len(r_own["lf_eq"]) == 32 and len(r_own["lf_raw"]) == 32
        small = os.path.join(tmp, "small.png"); Image.new("RGB", (300, 300)).save(small)
        assert extract_one((small, {"key": "s"}))["status"] == "error"
        # reanudacion: segunda ejecucion no anade nada y manifest distinto aborta
        run_dir = os.path.join(tmp, "run"); os.makedirs(run_dir)
        real_dir = os.path.join(tmp, "reals", "devX"); os.makedirs(real_dir); shutil.copy(own, os.path.join(real_dir, "a.png"))
        synth = os.path.join(tmp, "synth", "fam"); os.makedirs(synth); shutil.copy(plain, os.path.join(synth, "b.png"))
        with open(os.path.join(run_dir, "sample.json"), "w") as f:
            json.dump([{"key": "real/devX/a.png", "group": "real", "label": "real", "rel": "devX/a.png"},
                       {"key": "fam/fam/b.png", "group": "fam", "label": "fake", "rel": "fam/b.png"}], f)
        a = argparse.Namespace(run_dir=run_dir, real_dir=os.path.join(tmp, "reals"), synth_root=os.path.join(tmp, "synth"), registry_root=reg,
                               out=os.path.join(tmp, "out"), dresden_glob=None, workers=1)
        run(a)
        with open(os.path.join(a.out, "h1-features.jsonl")) as f:
            lines = f.readlines()
        assert len(lines) == 2 and json.loads(lines[0])["device"] == "devX"
        run(a)
        with open(os.path.join(a.out, "h1-features.jsonl")) as f:
            assert len(f.readlines()) == 2
        with open(os.path.join(run_dir, "sample.json"), "w") as f:
            json.dump([], f)
        try:
            run(a); raise AssertionError("debia abortar")
        except SystemExit as e:
            assert e.code == 2
        LOG.info("autotest OK (9 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--run-dir"); p.add_argument("--real-dir"); p.add_argument("--synth-root"); p.add_argument("--registry-root")
    p.add_argument("--out"); p.add_argument("--dresden-glob"); p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("run_dir", "real_dir", "synth_root", "registry_root", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    run(a)


if __name__ == "__main__":
    main()
