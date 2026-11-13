"""Corpus de reales nativos multi-dispositivo para la sonda v1. Criterios fijados ANTES de ver resultados:
- solo carpetas 'nat' (no 'flat'), JPEG, lado menor >= 2000 px (sin reescalar), sin PNG;
- como maximo --per-device fotos por dispositivo (muestreo con semilla);
- enlaces duros (cp -l) en OUT/<dispositivo>/ y manifiesto JSON escrito por el script. No se modifica ni reescala nada.
Uso: python3 build-native-reals.py --out DIR [--per-device 5] [--seed 20261111]"""
import argparse, hashlib, json, logging, os, random, shutil, sys
from PIL import Image

LOG = logging.getLogger("native-reals")
SOURCES = [
    ("/home/gustavo/dataset_test/registry-expansion-2026-11-04", None),
    ("/home/gustavo/dataset_test/vision-native-pilot", None),
    ("/home/gustavo/dataset_test/fotos_movil", "V2111"),
]
MIN_SIDE = 2000


def candidates():
    for root, fixed in SOURCES:
        for dp, _, names in os.walk(root):
            parts = dp.split(os.sep)
            if fixed:
                device = fixed
            elif parts[-1] == "nat":
                device = parts[-2]
            else:
                continue
            for n in sorted(names):
                if n.lower().endswith((".jpg", ".jpeg", ".png")):
                    yield device, os.path.join(dp, n)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--per-device", type=int, default=5); ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} ya existe: no se mezclan ejecuciones")
    by_dev, rejected = {}, {}
    for dev, path in candidates():
        with Image.open(path) as im:
            fmt, (w, h) = im.format, im.size
        why = "formato:" + str(fmt) if fmt != "JPEG" else ("lado<%d" % MIN_SIDE if min(w, h) < MIN_SIDE else None)
        if why:
            rejected.setdefault(f"{dev}|{why}", 0); rejected[f"{dev}|{why}"] += 1
            continue
        by_dev.setdefault(dev, []).append((path, w, h))
    rng = random.Random(a.seed)
    items = []
    for dev in sorted(by_dev):
        files = sorted(by_dev[dev]); rng.shuffle(files)
        for path, w, h in sorted(files[:a.per_device]):
            dst = os.path.join(a.out, dev, os.path.basename(path))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.link(path, dst)
            except OSError:
                shutil.copy2(path, dst)
            with open(path, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            items.append({"ruta": dst, "origen": path, "dispositivo": dev, "sha256": sha, "ancho": w, "alto": h, "recorte": False, "resize": False})
    man = {"criterios": {"carpeta": "nat", "formato": "JPEG", "lado_menor_min": MIN_SIDE, "por_dispositivo": a.per_device, "semilla": a.seed},
           "n": len(items), "dispositivos": len(by_dev), "rechazadas": rejected, "items": items}
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    LOG.info("%d fotos de %d dispositivos; rechazadas: %s", len(items), len(by_dev), rejected)


if __name__ == "__main__":
    main()
