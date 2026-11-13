#!/usr/bin/env python3
"""Prueba del decodificador (H1, seccion 9 de PREREGISTRO_H1.md): reales pasados por el VAE de Stable Diffusion.

Para cada real de run3 (205) toma el recorte 512 alineado a 8 (pixeles nativos, sin reescalar) y construye 4 brazos con el MISMO contenido:
  crop   el recorte original
  recon  recorte -> VAE (encode, media del latente, decode) de stabilityai/sd-vae-ft-mse
  blur   recorte con desenfoque gaussiano sigma 0.7 (control inespecifico: cualquier paso bajo)
  jpeg75 recorte recomprimido JPEG Q75 (control de recompresion)
Rasgos: res_spec y lf_eq con las funciones de h1-controls-extract.py (formato igual: arrays 8 bit sin compresion, salvo jpeg75).
Analisis (exploratorio salvo el criterio de la seccion 9):
  A) separabilidad crop vs recon por dispositivo (regresion logistica, 5 pliegues de dispositivo): AUC.
  B) modelo real-vs-IA entrenado con h1-controls-run1 brazo mod8 (solo reales de entrenamiento de otros pliegues): fraccion de pares en que
     recon puntua mas «IA» que su propio crop; lo mismo para blur y jpeg75 (referencia inespecifica).
Uso:
  python3 h1-vae-recon.py --run-dir RUN3 --real-dir DIR --controls-dir h1-controls-run1 --out OUT [--model ID] [--limit N]
  python3 h1-vae-recon.py --self-test
"""
import argparse, importlib.util, io, json, logging, os, sys, tempfile
import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import wilcoxon

LOG = logging.getLogger("h1-vae-recon")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


CTL = load_module("h1ctl", os.path.join(HERE, "h1-controls-extract.py"))
ANA = CTL.load_module("h1analyze", os.path.join(HERE, "h1-analyze.py"))
EX = CTL.EX
MODEL_ID = "stabilityai/sd-vae-ft-mse"
BLUR_SIGMA, JPEG_Q = 0.7, 75
ARMS = ("crop", "recon", "blur", "jpeg75")
SEED = ANA.SEED


class VaeCodec:
    def __init__(self, model_id):
        import torch
        from diffusers import AutoencoderKL
        torch.set_num_threads(2)
        self.torch = torch
        self.vae = AutoencoderKL.from_pretrained(model_id).eval()
        self.versions = {"torch": torch.__version__, "model": model_id}

    def __call__(self, img):
        torch = self.torch
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            out = self.vae.decode(self.vae.encode(x).latent_dist.mean).sample
        arr = ((out[0].permute(1, 2, 0).clamp(-1, 1).numpy() + 1.0) * 127.5).round().astype(np.uint8)
        return Image.fromarray(arr)


def make_arms(crop, codec):
    blur = crop.filter(ImageFilter.GaussianBlur(BLUR_SIGMA))
    jpeg, _ = CTL.jpeg_roundtrip(crop, JPEG_Q)
    return {"crop": crop, "recon": codec(crop), "blur": blur, "jpeg75": jpeg}


def process(item, codec):
    path, meta = item
    try:
        with Image.open(path) as src:
            img = src.convert("RGB")
        if min(img.size) < 512:
            raise ValueError("menor que 512")
        arms = {}
        for name, im in make_arms(CTL.aligned_crop(img), codec).items():
            lf, sp = CTL.features(im)
            arms[name] = {"lf_eq": lf, "res_spec": sp}
        return {**meta, "status": "ok", "arms": arms}
    except Exception as exc:
        return {**meta, "status": "error", "error": str(exc)[:200]}


def extract(a, codec):
    os.makedirs(a.out, exist_ok=True)
    items = [it for it in EX.build_items(argparse.Namespace(run_dir=a.run_dir, real_dir=a.real_dir, synth_root="", dresden_glob=None)) if it[1]["label"] == "real"]
    if a.limit:
        items = items[:a.limit]
    out_path = os.path.join(a.out, "h1-vae.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    manifest = {"script_sha256": EX.sha256_file(os.path.abspath(__file__)), "controls_sha256": EX.sha256_file(os.path.join(HERE, "h1-controls-extract.py")),
                "sample_sha256": EX.sha256_file(os.path.join(a.run_dir, "sample.json")), "n_items": len(items), "model": a.model, "blur_sigma": BLUR_SIGMA, "jpeg_q": JPEG_Q}
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            if json.load(f) != manifest:
                LOG.error("manifest distinto del existente: no se mezclan ejecuciones (borra --out)")
                sys.exit(2)
    else:
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
    todo = [it for it in items if it[1]["key"] not in done]
    LOG.info("%d reales, %d ya hechos, %d por hacer", len(items), len(done), len(todo))
    n_err = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, it in enumerate(todo, 1):
            rec = process(it, codec)
            out.write(json.dumps(rec) + "\n"); out.flush()
            n_err += rec["status"] != "ok"
            if i % 10 == 0:
                LOG.info("%d/%d (errores %d)", i, len(todo), n_err)
    LOG.info("fin: %d nuevos, errores %d", len(todo), n_err)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(l) for l in f if l.strip()) if r["status"] == "ok"]


def analyze(vae_dir, controls_dir, n_boot):
    vae = read_jsonl(os.path.join(vae_dir, "h1-vae.jsonl"))
    ctl = read_jsonl(os.path.join(controls_dir, "h1-controls.jsonl"))
    LOG.info("VAE: %d reales con 4 brazos; controles: %d imagenes", len(vae), len(ctl))
    dev = [r["device"] for r in vae]
    fold = ANA.device_folds(dev, np.random.default_rng(SEED))
    ctl_re = [r for r in ctl if r["label"] == "real"]
    ctl_ai = [r for r in ctl if r["label"] == "fake"]
    ctl_fold = ANA.device_folds([r["device"] for r in ctl_re], np.random.default_rng(SEED))
    # los dispositivos coinciden en ambos conjuntos: mismo reparto por dispositivo
    dev_fold = {d: f for d, f in zip(dev, fold)}
    ctl_fold = np.array([dev_fold[r["device"]] for r in ctl_re])
    out = {}
    for feat in ("res_spec", "lf_eq"):
        X = {arm: np.array([r["arms"][arm][feat] for r in vae]) for arm in ARMS}
        res = {}
        # A) crop vs recon (y vs los controles), por dispositivo
        for arm in ("recon", "blur", "jpeg75"):
            s_pos, s_neg = np.empty(len(vae)), np.empty(len(vae))
            for g in range(ANA.N_DEV_FOLDS):
                tr, te = fold != g, fold == g
                Xtr = np.vstack([X[arm][tr], X["crop"][tr]]); ytr = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
                m = ANA.model().fit(Xtr, ytr)
                s_pos[te], s_neg[te] = m.decision_function(X[arm][te]), m.decision_function(X["crop"][te])
            rng = np.random.default_rng(SEED); ud = sorted(set(dev)); idx = {d: np.where(np.array(dev) == d)[0] for d in ud}
            boots = []
            for _ in range(n_boot):
                sel = np.concatenate([idx[ud[i]] for i in rng.choice(len(ud), len(ud))])
                boots.append(ANA.auc(s_pos[sel], s_neg[sel]))
            res[f"A_{arm}_vs_crop"] = {"auc": ANA.auc(s_pos, s_neg), "ic95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
            LOG.info("A) %-8s %-6s vs crop (por dispositivo) AUC=%.3f [%.3f, %.3f]", feat, arm, res[f"A_{arm}_vs_crop"]["auc"], *res[f"A_{arm}_vs_crop"]["ic95"])
        # B) modelo real-vs-IA
        score = {arm: np.empty(len(vae)) for arm in ARMS}
        X_ai = np.array([r["arms"]["mod8"][feat] for r in ctl_ai]); X_re = np.array([r["arms"]["mod8"][feat] for r in ctl_re])
        for g in range(ANA.N_DEV_FOLDS):
            tr = ctl_fold != g
            m = ANA.model().fit(np.vstack([X_ai, X_re[tr]]), np.r_[np.ones(len(X_ai)), np.zeros(tr.sum())])
            for arm in ARMS:
                score[arm][fold == g] = m.decision_function(X[arm][fold == g])
        for arm in ("recon", "blur", "jpeg75"):
            d = score[arm] - score["crop"]
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            res[f"B_{arm}"] = {"frac_mas_IA_que_su_crop": float(np.mean(d > 0)), "mediana_desplazamiento": float(np.median(d)), "wilcoxon_p": p}
            LOG.info("B) %-8s %-6s puntua mas IA que su crop en %.1f%% de pares; mediana desplazamiento %.2f; p=%.2g", feat, arm, 100 * np.mean(d > 0), np.median(d), p)
        res["B_crop_frac_sobre_umbral_0"] = float(np.mean(score["crop"] > 0))
        res["B_recon_frac_sobre_umbral_0"] = float(np.mean(score["recon"] > 0))
        LOG.info("B) %-8s crop clasificado IA (score>0): %.1f%%; recon clasificado IA: %.1f%%", feat, 100 * res["B_crop_frac_sobre_umbral_0"], 100 * res["B_recon_frac_sobre_umbral_0"])
        out[feat] = res
    out["veredicto"] = verdict(out)
    LOG.info("VEREDICTO decodificador: %s", out["veredicto"])
    return out


def verdict(out):
    r = out["res_spec"]
    sep = r["A_recon_vs_crop"]["ic95"][0] > 0.90
    shift = r["B_recon"]["frac_mas_IA_que_su_crop"] >= 0.80 and r["B_recon"]["wilcoxon_p"] < 0.001
    beats_blur = r["B_recon"]["mediana_desplazamiento"] > max(r["B_blur"]["mediana_desplazamiento"], r["B_jpeg75"]["mediana_desplazamiento"])
    if sep and shift and beats_blur:
        return "huella de decodificador apoyada"
    if not sep or not shift:
        return "no apoyada"
    return "indeterminada (no supera a los controles inespecificos)"


def self_test():
    import shutil
    rng = np.random.default_rng(4)
    tmp = tempfile.mkdtemp()
    try:
        yy, xx = np.mgrid[0:700, 0:800]
        base = 0.5 + 0.2 * np.sin(xx / 60.0) * np.cos(yy / 50.0)
        real_dir = os.path.join(tmp, "reals"); run_dir = os.path.join(tmp, "run"); os.makedirs(run_dir)
        sample = []
        for d in range(10):
            os.makedirs(os.path.join(real_dir, f"dev{d}"))
            for i in range(5):
                im = np.clip(base[..., None] + rng.normal(0, 0.02, (700, 800, 3)), 0, 1)
                Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(real_dir, f"dev{d}", f"{i}.jpg"), quality=92)
                sample.append({"key": f"real/dev{d}/{i}.jpg", "group": "real", "label": "real", "rel": f"dev{d}/{i}.jpg"})
        with open(os.path.join(run_dir, "sample.json"), "w") as f:
            json.dump(sample, f)
        fake_codec = lambda img: img.filter(ImageFilter.GaussianBlur(2.0))  # decodificador simulado: paso bajo fuerte
        a = argparse.Namespace(run_dir=run_dir, real_dir=real_dir, controls_dir=None, out=os.path.join(tmp, "out"), model="mock", limit=0)
        extract(a, fake_codec)
        extract(a, fake_codec)
        recs = read_jsonl(os.path.join(a.out, "h1-vae.jsonl"))
        assert len(recs) == 50 and set(recs[0]["arms"]) == set(ARMS) and len(recs[0]["arms"]["recon"]["res_spec"]) == 64
        assert np.mean(recs[0]["arms"]["recon"]["res_spec"][-10:]) < np.mean(recs[0]["arms"]["crop"]["res_spec"][-10:])
        # controles sinteticos: IA = reales con paso bajo moderado, para que el modelo real-vs-IA exista
        cdir = os.path.join(tmp, "ctl"); os.makedirs(cdir)
        with open(os.path.join(cdir, "h1-controls.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps({"key": r["key"], "group": "real", "label": "real", "device": r["device"], "status": "ok", "arms": {"mod8": r["arms"]["crop"]}}) + "\n")
                f.write(json.dumps({"key": "ai" + r["key"], "group": "fam", "label": "fake", "device": None, "status": "ok", "arms": {"mod8": r["arms"]["recon"]}}) + "\n")
        out = analyze(a.out, cdir, 30)
        assert out["res_spec"]["A_recon_vs_crop"]["auc"] > 0.95 and out["res_spec"]["B_recon"]["frac_mas_IA_que_su_crop"] > 0.9, out["res_spec"]
        assert out["veredicto"] in ("huella de decodificador apoyada", "indeterminada (no supera a los controles inespecificos)")
        assert verdict({"res_spec": {"A_recon_vs_crop": {"ic95": [0.5, 0.6]}, "B_recon": {"frac_mas_IA_que_su_crop": 0.5, "wilcoxon_p": 0.5, "mediana_desplazamiento": 0}, "B_blur": {"mediana_desplazamiento": 0}, "B_jpeg75": {"mediana_desplazamiento": 0}}}) == "no apoyada"
        LOG.info("autotest OK (5 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--run-dir"); p.add_argument("--real-dir"); p.add_argument("--controls-dir"); p.add_argument("--out")
    p.add_argument("--model", default=MODEL_ID); p.add_argument("--limit", type=int, default=0); p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--analyze-only", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("run_dir", "real_dir", "controls_dir", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    if not a.analyze_only:
        extract(a, VaeCodec(a.model))
    result = analyze(a.out, a.controls_dir, a.n_boot)
    with open(os.path.join(a.out, "h1-vae-analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h1-vae-analysis.json"))


if __name__ == "__main__":
    main()
