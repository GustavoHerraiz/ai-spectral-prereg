#!/usr/bin/env python3
"""Inspeccion de solo lectura: formato, EXIF, calidad JPEG estimada y huella de bloques 8x8
de una muestra de 0_real y 1_fake por generador. No escribe nada salvo --out (JSON opcional)."""
import argparse
import json
import logging
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, JpegImagePlugin

LOG = logging.getLogger("inspect-format-bias")
IMG_EXT = (".png", ".jpg", ".jpeg")
UNI_AUC_MAX_DEV = 0.15  # una variable de formato/tamano no debe separar las clases mejor que esto
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def estimate_quality(qtable):
    """Calidad libjpeg que mejor reproduce la tabla de luminancia (None si no cuadra)."""
    best_q, best_err = None, None
    q = np.asarray(qtable, dtype=float)
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        cand = np.clip((STD_LUMA * scale + 50) // 100, 1, 255)
        err = float(np.abs(cand - q).mean())
        if best_err is None or err < best_err:
            best_q, best_err = quality, err
    return best_q if best_err is not None and best_err < 1.0 else None


def blockiness(img):
    """Cociente entre salto medio en fronteras 8x8 y en el resto; >1 sugiere historial JPEG."""
    g = np.asarray(img.convert("L"), dtype=np.float64)
    h, w = g.shape
    dx = np.abs(np.diff(g[:, : w - w % 8], axis=1))
    cols = np.arange(dx.shape[1])
    on = dx[:, cols % 8 == 7].mean()
    off = dx[:, cols % 8 != 7].mean()
    return float(on / (off + 1e-9))


def inspect_file(path):
    with Image.open(path) as im:
        info = {
            "ext": os.path.splitext(path)[1].lower(),
            "format": im.format,
            "mode": im.mode,
            "size": im.size,
            "bytes": os.path.getsize(path),
            "exif": bool(len(im.getexif())),
            "icc": "icc_profile" in im.info,
            "png_text": bool(getattr(im, "text", None)) if im.format == "PNG" else False,
        }
        if im.format == "JPEG":
            tables = getattr(im, "quantization", {}) or {}
            info["jpeg_quality"] = estimate_quality(tables[0]) if 0 in tables else None
            info["subsampling"] = JpegImagePlugin.get_sampling(im)
        im.load()
        info["blockiness"] = blockiness(im)
    return info


def list_class(gdir, cls):
    files = []
    for dirpath, _, names in os.walk(gdir):
        if os.path.basename(dirpath) == cls:
            files += [os.path.join(dirpath, n) for n in sorted(names) if n.lower().endswith(IMG_EXT)]
    return files


def summarize(infos):
    if not infos:
        return None
    fmt = Counter(i["format"] for i in infos)
    ext_mismatch = sum(1 for i in infos if (i["ext"] == ".png") != (i["format"] == "PNG"))
    q = [i["jpeg_quality"] for i in infos if i.get("jpeg_quality") is not None]
    sizes = Counter(i["size"] for i in infos)
    return {
        "n": len(infos),
        "formatos": dict(fmt),
        "extension_no_coincide_con_formato": ext_mismatch,
        "exif_pct": round(100 * np.mean([i["exif"] for i in infos]), 1),
        "icc_pct": round(100 * np.mean([i["icc"] for i in infos]), 1),
        "jpeg_calidad_estimada": dict(Counter(q).most_common(5)) if q else None,
        "submuestreo_jpeg": dict(Counter(i.get("subsampling") for i in infos if "subsampling" in i)) or None,
        "bytes_por_pixel_mediana": round(float(np.median([i["bytes"] / (i["size"][0] * i["size"][1]) for i in infos])), 3),
        "tamanos_top3": {f"{k[0]}x{k[1]}": v for k, v in sizes.most_common(3)},
        "blockiness_mediana": round(float(np.median([i["blockiness"] for i in infos])), 4),
    }


def uni_auc(real_vals, fake_vals):
    """AUC univariante (prob. de que un valor fake supere a uno real; empates = 0.5)."""
    r = np.sort(np.asarray(real_vals, dtype=float))
    f = np.asarray(fake_vals, dtype=float)
    less = np.searchsorted(r, f, side="left")
    leq = np.searchsorted(r, f, side="right")
    return float(((less + leq) / 2).sum() / (len(r) * len(f)))


def size_bytes_auc(real_infos, fake_infos):
    """AUC univariante por clase de variables que no deberian separar real de fake."""
    feats = {
        "bytes": lambda i: i["bytes"],
        "bytes_por_pixel": lambda i: i["bytes"] / (i["size"][0] * i["size"][1]),
        "ancho": lambda i: i["size"][0],
        "alto": lambda i: i["size"][1],
        "area": lambda i: i["size"][0] * i["size"][1],
        "blockiness": lambda i: i["blockiness"],
    }
    out = {}
    for name, fn in feats.items():
        out[name] = round(uni_auc([fn(i) for i in real_infos], [fn(i) for i in fake_infos]), 4)
    return out


def verdict(real, fake):
    flags = []
    if real["formatos"] != fake["formatos"]:
        flags.append(f"formatos distintos: real={real['formatos']} fake={fake['formatos']}")
    if abs(real["blockiness_mediana"] - fake["blockiness_mediana"]) > 0.02:
        flags.append(f"blockiness real={real['blockiness_mediana']} fake={fake['blockiness_mediana']}")
    if abs(real["bytes_por_pixel_mediana"] - fake["bytes_por_pixel_mediana"]) / max(real["bytes_por_pixel_mediana"], 1e-9) > 0.5:
        flags.append("bytes/pixel difieren >50%")
    rs, fs = set(real["tamanos_top3"]), set(fake["tamanos_top3"])
    if len(rs & fs) / len(rs | fs) < 0.5:  # Jaccard de los tamanos dominantes; ignora diferencias de reparto
        flags.append(f"tamanos distintos: real={real['tamanos_top3']} fake={fake['tamanos_top3']}")
    if abs(real["exif_pct"] - fake["exif_pct"]) > 20:
        flags.append(f"EXIF real={real['exif_pct']}% fake={fake['exif_pct']}%")
    for name, a in (real.get("auc_univariante") or {}).items():
        if abs(a - 0.5) > UNI_AUC_MAX_DEV:
            flags.append(f"{name} separa real/fake por si solo: AUC univariante={a} (|AUC-0.5| > {UNI_AUC_MAX_DEV})")
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/gustavo/dataset_test/cnn_synth_testset")
    ap.add_argument("--n", type=int, default=200, help="imagenes por generador y clase")
    ap.add_argument("--out", default=None, help="JSON de salida opcional")
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    rng = np.random.default_rng(args.seed)
    report, biased = {}, []
    for gen in sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))):
        gdir = os.path.join(args.root, gen)
        per, all_infos = {}, {}
        for cls in ("0_real", "1_fake"):
            files = list_class(gdir, cls)
            if len(files) > args.n:
                files = [files[i] for i in sorted(rng.choice(len(files), args.n, replace=False))]
            infos = []
            for f in files:
                try:
                    infos.append(inspect_file(f))
                except Exception as exc:  # archivo ilegible
                    LOG.warning("ilegible %s: %s", f, exc)
            per[cls] = summarize(infos)
            all_infos[cls] = infos
        report[gen] = per
        LOG.info("== %s", gen)
        for cls in ("0_real", "1_fake"):
            LOG.info("  %s %s", cls, json.dumps(per[cls], ensure_ascii=False))
        if per["0_real"] and per["1_fake"]:
            per["0_real"]["auc_univariante"] = size_bytes_auc(all_infos["0_real"], all_infos["1_fake"])
            flags = verdict(per["0_real"], per["1_fake"])
            report[gen]["alertas"] = flags
            if flags:
                biased.append(gen)
            LOG.info("  ALERTAS: %s", flags if flags else "ninguna")
    LOG.info("\nGeneradores con alertas de sesgo (%d/%d): %s", len(biased), len(report), biased)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
