#!/usr/bin/env python3
"""Archiva los resultados de H1 (solo en el servidor) dentro del repo con manifiesto SHA-256. No modifica los originales.
Copia *.json, *.jsonl, *.log y manifest*.json de los directorios dados a l1-manisha2025/results-archive/<nombre>/ y escribe
MANIFEST.sha256 (hash de cada copia y del original) y ARCHIVE_INFO.json (commit, fecha, tamanos). Omite ficheros > --max-mb.
Uso:
  python3 archive-results.py --results-root l1-manisha2025/results --dest l1-manisha2025/results-archive --names h1-extract-run1,h1-controls-run1,...
  python3 archive-results.py --self-test
"""
import argparse, datetime, hashlib, json, logging, os, shutil, subprocess, sys, tempfile

LOG = logging.getLogger("archive-results")
EXT = (".json", ".jsonl", ".log", ".sha256")
DEFAULT = ["h1-extract-run1", "h1-controls-run1", "h1-loco-run1", "h1-native-run1", "h1-scale-run1", "h1-resample-run1", "h1-owngen-run1", "h1-chain-run1", "h1r-explore-run1", "h2-e0-run1", "h2-e1-run1", "h2-e2-run1", "h1-vae-auc.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def archive(root, dest, names, max_mb):
    os.makedirs(dest, exist_ok=True)
    lines, info, skipped = [], [], []
    for name in names:
        src = os.path.join(root, name)
        files = [src] if os.path.isfile(src) else [os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs if f.lower().endswith(EXT)] if os.path.isdir(src) else []
        if not files:
            LOG.warning("no existe o vacio: %s", src); skipped.append(name); continue
        for f in sorted(files):
            if os.path.getsize(f) > max_mb * 1e6:
                LOG.warning("omitido por tamano: %s", f); skipped.append(f); continue
            rel = os.path.relpath(f, root) if os.path.isdir(src) or os.path.isfile(src) else name
            out = os.path.join(dest, rel); os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(f, out)
            assert sha(f) == sha(out), f"copia distinta: {f}"
            lines.append(f"{sha(out)}  {os.path.relpath(out, dest)}"); info.append({"file": os.path.relpath(out, dest), "bytes": os.path.getsize(out)})
    with open(os.path.join(dest, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(lines)) + "\n")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    with open(os.path.join(dest, "ARCHIVE_INFO.json"), "w", encoding="utf-8") as f:
        json.dump({"fecha_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "commit_codigo": commit, "ficheros": info, "omitidos_o_ausentes": skipped}, f, indent=1)
    LOG.info("archivados %d ficheros; omitidos/ausentes: %s", len(lines), skipped)
    return len(lines), skipped


def self_test():
    tmp = tempfile.mkdtemp()
    try:
        r, d = os.path.join(tmp, "r"), os.path.join(tmp, "d")
        os.makedirs(os.path.join(r, "run1")); open(os.path.join(r, "run1", "a.json"), "w").write("{}"); open(os.path.join(r, "run1", "b.npz"), "w").write("x"); open(os.path.join(r, "solo.json"), "w").write("[]")
        n, sk = archive(r, d, ["run1", "solo.json", "falta"], 1)
        assert n == 2 and sk == ["falta"], (n, sk)
        assert not os.path.exists(os.path.join(d, "run1", "b.npz")) and "a.json" in open(os.path.join(d, "MANIFEST.sha256")).read()
        LOG.info("autotest OK (2 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true"); p.add_argument("--results-root"); p.add_argument("--dest")
    p.add_argument("--names", default=",".join(DEFAULT)); p.add_argument("--max-mb", type=float, default=25)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if not a.results_root or not a.dest:
        p.error("faltan --results-root y --dest")
    archive(a.results_root, a.dest, [n for n in a.names.split(",") if n], a.max_mb)


if __name__ == "__main__":
    main()
