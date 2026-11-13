#!/usr/bin/env python3
"""Construye el corpus L2 sin remuestreo: crop central 512x512, NUNCA resize, mismo formato de salida para ambas clases.

ESQUELETO: no se ha ejecutado. Por defecto solo hace un recuento (--dry-run implicito); escribe solo con --execute.

Reglas (preregistro L2):
  R1  Prohibido reescalar. Imagen < 512 en algun lado -> se descarta y se registra. == 512 -> se copia. > 512 -> crop central.
  R2  El origen del crop se redondea hacia abajo a multiplo de 16, para conservar la rejilla 8x8 de JPEG y la de 8/16 px
      de los autoencoders latentes. Se registra el desplazamiento.
  R3  Salida identica para ambas clases: PNG RGB 8 bit sin metadatos. Decodificar un JPEG a PNG no altera pixeles,
      pero el historial de compresion de las reales SIGUE EN LOS PIXELES: se declara, no se elimina.
  R4  Se rechazan (no se convierten) imagenes con alfa, 16 bit o modos no RGB/L8. L se descarta tambien.
  R5  No se aplica rotacion EXIF; se registra la orientacion.
  R6  Una imagen de origen -> como maximo un crop (sin pseudorreplicacion). Duplicados por SHA-256 se descartan.
  R7  Seleccion determinista: orden por SHA-256 del origen + semilla; tope --per-source por fuente.
  R8  Manifiesto JSON escrito por este script, con hash de origen y de salida.

Configuracion: JSON con la lista de fuentes (--config). Cada fuente:
  {"name": "...", "label": "real"|"fake", "family": "...", "root": "/ruta", "glob": "**/*.png",
   "origin": "texto libre con procedencia y licencia", "native_generation": true|false|null}
`native_generation` es una declaracion del autor (p. ej. false si el generador salio a 1024 y se recorta). null = desconocido
y la fuente queda marcada 'procedencia_no_verificada' en el manifiesto.
"""
import argparse
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
from glob import glob

import numpy as np
from PIL import Image

LOG = logging.getLogger("build-l2-corpus")
SIZE = 512
GRID = 16
SEED = 20261111
OUT_FORMAT = "PNG"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def crop_box(w, h, size=SIZE, grid=GRID):
    """Crop central con origen multiplo de `grid`. Devuelve (left, top, right, bottom)."""
    left = ((w - size) // 2) // grid * grid
    top = ((h - size) // 2) // grid * grid
    return left, top, left + size, top + size


def load_rgb8(path):
    """Devuelve (imagen RGB, info) o (None, motivo). Sin conversion de profundidad ni de alfa."""
    with Image.open(path) as im:
        info = {"source_format": im.format, "source_mode": im.mode, "width": im.size[0], "height": im.size[1],
                "exif_orientation": int(im.getexif().get(274, 1)), "icc_profile": "icc_profile" in im.info}
        if im.mode != "RGB":
            return None, {**info, "reason": f"modo_no_admitido:{im.mode}"}
        if im.size[0] < SIZE or im.size[1] < SIZE:
            return None, {**info, "reason": "menor_que_512_sin_resize"}
        im.load()
        return im.copy(), info


def process_source(src, per_source, out_dir, rng_seed, execute):
    """Devuelve (entradas_manifiesto, descartes) de una fuente."""
    files = sorted(glob(os.path.join(src["root"], src["glob"]), recursive=True))
    hashed = []
    for f in files:
        if os.path.isfile(f) and not os.path.islink(f):
            hashed.append((sha256_file(f), f))
    # orden determinista pseudoaleatorio, independiente del orden del sistema de ficheros
    hashed.sort(key=lambda t: sha256_bytes((t[0] + str(rng_seed)).encode()))
    entries, dropped, seen = [], {}, set()
    for src_hash, path in hashed:
        if len(entries) >= per_source:
            break
        if src_hash in seen:
            dropped["duplicado"] = dropped.get("duplicado", 0) + 1
            continue
        seen.add(src_hash)
        try:
            img, info = load_rgb8(path)
        except Exception as exc:  # archivo corrupto o formato no soportado
            LOG.warning("ilegible %s: %s", path, exc)
            dropped["ilegible"] = dropped.get("ilegible", 0) + 1
            continue
        if img is None:
            dropped[info["reason"]] = dropped.get(info["reason"], 0) + 1
            continue
        w, h = info["width"], info["height"]
        cropped = (w, h) != (SIZE, SIZE)
        box = crop_box(w, h) if cropped else (0, 0, SIZE, SIZE)
        out = img.crop(box) if cropped else img
        assert out.size == (SIZE, SIZE), "invariante: salida 512x512 sin resize"
        buf = io.BytesIO()
        out.save(buf, format=OUT_FORMAT)  # sin metadatos, sin optimizar variantes de color
        data = buf.getvalue()
        out_hash = sha256_bytes(data)
        rel = os.path.join(src["label"], src["family"], f"{out_hash[:16]}.png")
        if execute:
            dest = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(data)
        entries.append({
            "path": rel, "sha256": out_hash, "label": src["label"], "family": src["family"],
            "source_name": src["name"], "origin": src["origin"],
            "native_generation_declared": src.get("native_generation"),
            "procedencia_no_verificada": src.get("native_generation") is None,
            "source_path": path, "source_sha256": src_hash,
            "source_format": info["source_format"], "source_mode": info["source_mode"],
            "original_resolution": [w, h], "cropped": cropped,
            "crop_box": list(box) if cropped else None, "crop_size": [SIZE, SIZE] if cropped else None,
            "resized": False, "exif_orientation_ignored": info["exif_orientation"], "icc_profile_dropped": info["icc_profile"],
            "output_format": OUT_FORMAT,
        })
    return entries, dropped


def build(config, out_dir, per_source, seed, execute):
    manifest = {"version": 1, "seed": seed, "size": SIZE, "grid": GRID, "output_format": OUT_FORMAT,
                "rules": [l.strip() for l in __doc__.splitlines() if l.strip().startswith("R")],
                "executed": execute, "sources": {}, "entries": []}
    for src in config:
        entries, dropped = process_source(src, per_source, out_dir, seed, execute)
        manifest["sources"][src["name"]] = {"kept": len(entries), "dropped": dropped, "origin": src["origin"]}
        manifest["entries"] += entries
        LOG.info("%s: %d conservadas, descartes=%s", src["name"], len(entries), dropped)
    # un mismo SHA de salida en dos clases invalidaria el corpus
    by_hash = {}
    for e in manifest["entries"]:
        by_hash.setdefault(e["sha256"], set()).add(e["label"])
    if any(len(v) > 1 for v in by_hash.values()):
        raise RuntimeError("imagen de salida identica en reales y fakes")
    if execute:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def self_test():
    """Autotest sintetico (pendiente de ejecutar): sin resize, offset multiplo de 16, mismo formato, manifiesto completo."""
    ok = []
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(SEED)
        for name, size, fmt in (("a", (1024, 768), "PNG"), ("b", (512, 512), "PNG"), ("c", (700, 600), "JPEG"), ("d", (400, 600), "PNG")):
            os.makedirs(os.path.join(tmp, "in", name))
            arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(tmp, "in", name, f"x.{fmt.lower()}"), format=fmt)
        cfg = [{"name": n, "label": "real" if n in "ab" else "fake", "family": n, "root": os.path.join(tmp, "in", n),
                "glob": "*.*", "origin": "sintetico", "native_generation": True} for n in "abcd"]
        m = build(cfg, os.path.join(tmp, "out"), 10, SEED, True)
        ok.append(len(m["entries"]) == 3 and m["sources"]["d"]["dropped"].get("menor_que_512_sin_resize") == 1)
        ok.append(all(tuple(e["crop_box"][:2]) == (e["crop_box"][0] // GRID * GRID, e["crop_box"][1] // GRID * GRID)
                      for e in m["entries"] if e["cropped"]))
        a = next(e for e in m["entries"] if e["family"] == "a")
        src = np.asarray(Image.open(os.path.join(tmp, "in", "a", "x.png")))
        got = np.asarray(Image.open(os.path.join(tmp, "out", a["path"])))
        l, t, r, b = a["crop_box"]
        ok.append(np.array_equal(src[t:b, l:r], got))  # pixeles identicos al recorte: no hubo remuestreo
        ok.append(all(Image.open(os.path.join(tmp, "out", e["path"])).format == "PNG" for e in m["entries"]))
        ok.append(os.path.exists(os.path.join(tmp, "out", "manifest.json")))
    LOG.info("autotest: %d/%d correctos %s", sum(ok), len(ok), ok)
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON con la lista de fuentes")
    ap.add_argument("--out", default=None, help="directorio de salida (corpus + manifest.json)")
    ap.add_argument("--per-source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--execute", action="store_true", help="escribir imagenes y manifiesto; sin esto solo cuenta")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        return self_test()
    if not args.config or (args.execute and not args.out):
        ap.error("--config es obligatorio; --out es obligatorio con --execute")
    with open(args.config) as fh:
        config = json.load(fh)
    build(config, args.out or "", args.per_source, args.seed, args.execute)
    return 0


if __name__ == "__main__":
    sys.exit(main())
