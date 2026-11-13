#!/usr/bin/env python3
"""AUC de res_spec y lf_eq para reales originales, reales reconstruidos con VAE (hechos por el usuario) e IA Synthbuster.

A) VAE > Original (pares por nombre, validacion por dispositivo). B) IA > VAE. C) IA > Original (linea base).
Con --orig-dir: originales = imagenes de ese directorio ordenadas alfabeticamente; real_vae_i corresponde a la i-esima (verificado por miniatura).
B y C se calculan de dos formas (misma IA y mismo AUC macro sobre familias, una familia fuera en el entrenamiento):
  fuera de dominio: reales de entrenamiento = los 205 reales de run3 (h1-controls-run1, mod8)
  en dominio: reales de entrenamiento = las otras originales (5 pliegues; la VAE de una original nunca se entrena con su original)
IA: h1-controls-run1, brazo mod8. IC95: bootstrap de dispositivos (reales) y de imagenes dentro de cada familia (IA).
Uso: python3 h1-vae-auc.py --vae-dir DIR --orig-dir DIR --controls-dir DIR [--run-dir RUN3 --real-dir DIR] [--out FILE] [--n-boot 2000]
     python3 h1-vae-auc.py --self-test
"""
import argparse, collections, glob, json, logging, os, sys, tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = logging.getLogger("h1-vae-auc")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


CTL = load_module("h1ctl", os.path.join(HERE, "h1-controls-extract.py"))
ANA = CTL.load_module("h1analyze", os.path.join(HERE, "h1-analyze.py"))
SEED = ANA.SEED
FEATS = ("res_spec", "lf_eq")
EXT = (".png", ".jpg", ".jpeg")


def list_images(root):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(EXT))


def pair_up(sample, real_dir, vae_dir):
    by_stem = collections.defaultdict(list)
    for s in sample:
        if s["label"] == "real":
            by_stem[os.path.splitext(os.path.basename(s["rel"]))[0]].append(s)
    pairs, unmatched = [], []
    for p in list_images(vae_dir):
        stem = os.path.splitext(os.path.basename(p))[0]
        cands = by_stem.get(stem) or by_stem.get(stem.rsplit("_", 1)[0], [])
        if len(cands) != 1:
            unmatched.append((p, len(cands))); continue
        pairs.append((cands[0], os.path.join(real_dir, cands[0]["rel"]), p))
    return pairs, unmatched


def center_crop(img, size):
    w, h = img.size
    left, top = (w - size[0]) // 2, (h - size[1]) // 2
    return img.crop((left, top, left + size[0], top + size[1]))


def feats_of(path, like_size=None):
    """Con like_size (tamano de la VAE menor que el original) se usa el recorte central de ese tamano, como load_rgb_native512."""
    with Image.open(path) as im:
        img = im.convert("RGB")
    if min(img.size) < 512:
        raise ValueError(f"{path}: lado menor que 512")
    if like_size and like_size != img.size:
        img = center_crop(img, like_size)
    lf, sp = CTL.features(CTL.aligned_crop(img))
    return {"lf_eq": lf, "res_spec": sp}, img.size


def thumb(img, n=24):
    g = np.asarray(img.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float64).ravel()
    g = g - g.mean()
    return g / (np.linalg.norm(g) + 1e-12)


def pair_by_order(orig_dir, vae_dir):
    """real_vae_i <-> i-esima imagen de orig_dir en orden alfabetico. Verifica con miniaturas; aborta si algun par no se parece."""
    origs = sorted(p for p in glob.glob(os.path.join(orig_dir, "*")) if p.lower().endswith(EXT))
    vaes = list_images(vae_dir)
    LOG.info("originales en %s: %d (formatos %s); VAE: %d", orig_dir, len(origs), dict(collections.Counter(os.path.splitext(p)[1].lower() for p in origs)), len(vaes))
    if len(origs) < len(vaes):
        LOG.error("hay menos originales que VAE"); sys.exit(2)
    pairs, corr = [], []
    for i, pv in enumerate(vaes):
        with Image.open(pv) as v, Image.open(origs[i]) as o:
            corr.append(float(thumb(v.convert("RGB")) @ thumb(o.convert("RGB"))))
        pairs.append(({"rel": f"img{i:03d}"}, origs[i], pv))
    LOG.info("verificacion por miniatura: correlacion min=%.3f mediana=%.3f", min(corr), float(np.median(corr)))
    if min(corr) < 0.90:
        LOG.error("algun par no se parece (min %.3f): el orden no coincide; no se calcula nada", min(corr)); sys.exit(2)
    return pairs


def pair_by_content(sample, real_dir, vae_paths, min_corr=0.90, min_gap=0.02):
    """Empareja cada VAE con el real de run3 de mayor correlacion de miniatura (recorte central o imagen completa)."""
    reals = [s for s in sample if s["label"] == "real"]
    cache, pairs, unmatched = {}, [], []
    for pv in vae_paths:
        with Image.open(pv) as im:
            v = im.convert("RGB")
        tv = thumb(v)
        scores = []
        for s in reals:
            key = (s["rel"], v.size)
            if key not in cache:
                with Image.open(os.path.join(real_dir, s["rel"])) as im:
                    o = im.convert("RGB")
                cache[key] = (thumb(center_crop(o, v.size)) if min(o.size) >= max(v.size) and o.size != v.size else None, thumb(o))
            tc, tw = cache[key]
            scores.append(max(float(tv @ tw), float(tv @ tc) if tc is not None else -1))
        order = np.argsort(scores)[::-1]
        best, second = scores[order[0]], scores[order[1]]
        if best >= min_corr and best - second >= min_gap:
            pairs.append((reals[order[0]], os.path.join(real_dir, reals[order[0]]["rel"]), pv))
        else:
            unmatched.append((pv, round(best, 3), round(second, 3)))
    return pairs, unmatched


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(l) for l in f if l.strip()) if r["status"] == "ok"]


def ci(vals):
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def scenario_A(Xo, Xv, dev, n_boot):
    fold = ANA.device_folds(dev, np.random.default_rng(SEED))
    nf = min(ANA.N_DEV_FOLDS, len(set(dev)))
    fold = fold % nf
    sv, so = np.empty(len(dev)), np.empty(len(dev))
    for g in range(nf):
        tr, te = fold != g, fold == g
        m = ANA.model().fit(np.vstack([Xv[tr], Xo[tr]]), np.r_[np.ones(tr.sum()), np.zeros(tr.sum())])
        sv[te], so[te] = m.decision_function(Xv[te]), m.decision_function(Xo[te])
    rng = np.random.default_rng(SEED); ud = sorted(set(dev)); idx = {d: np.where(np.array(dev) == d)[0] for d in ud}
    boots = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[ud[i]] for i in rng.choice(len(ud), len(ud))])
        boots.append(ANA.auc(sv[sel], so[sel]))
    return ANA.auc(sv, so), ci(boots)


def scenario_BC(Xo, Xv, dev, X_ai, fam_ai, X_re_train, n_boot, indomain=False):
    fams = sorted(set(fam_ai))
    S = {}
    fold = ANA.device_folds(dev, np.random.default_rng(SEED)) if indomain else None
    for f in fams:
        tr_ai = fam_ai != f
        if not indomain:
            m = ANA.model().fit(np.vstack([X_ai[tr_ai], X_re_train]), np.r_[np.ones(tr_ai.sum()), np.zeros(len(X_re_train))])
            S[f] = {"ai": m.decision_function(X_ai[fam_ai == f]), "o": m.decision_function(Xo), "v": m.decision_function(Xv)}
            continue
        so, sv, sa = np.empty(len(Xo)), np.empty(len(Xo)), []
        for g in range(ANA.N_DEV_FOLDS):
            tr = fold != g
            m = ANA.model().fit(np.vstack([X_ai[tr_ai], Xo[tr]]), np.r_[np.ones(tr_ai.sum()), np.zeros(tr.sum())])
            so[~tr], sv[~tr] = m.decision_function(Xo[~tr]), m.decision_function(Xv[~tr]); sa.append(m.decision_function(X_ai[fam_ai == f]))
        S[f] = {"ai": np.mean(sa, axis=0), "o": so, "v": sv}
    rng = np.random.default_rng(SEED); ud = sorted(set(dev)); idx = {d: np.where(np.array(dev) == d)[0] for d in ud}
    out = {}
    for key, name in (("v", "B_IA_vs_VAE"), ("o", "C_IA_vs_original")):
        point = np.mean([ANA.auc(S[f]["ai"], S[f][key]) for f in fams])
        boots = []
        for _ in range(n_boot):
            sel = np.concatenate([idx[ud[i]] for i in rng.choice(len(ud), len(ud))])
            boots.append(np.mean([ANA.auc(rng.choice(S[f]["ai"], len(S[f]["ai"])), S[f][key][sel]) for f in fams]))
        out[name] = (float(point), ci(boots), {f: float(ANA.auc(S[f]["ai"], S[f][key])) for f in fams})
    return out


def run(a):
    sample = []
    if a.run_dir:
        with open(os.path.join(a.run_dir, "sample.json"), encoding="utf-8") as f:
            sample = json.load(f)
    if a.orig_dir:
        pairs = pair_by_order(a.orig_dir, a.vae_dir)
        unmatched = []
    else:
        pairs, unmatched = pair_up(sample, a.real_dir, a.vae_dir)
        LOG.info("VAE: %d imagenes emparejadas por nombre, %d sin emparejar", len(pairs), len(unmatched))
    if len(pairs) < 10 and not a.orig_dir:
        pairs, unmatched = pair_by_content(sample, a.real_dir, list_images(a.vae_dir))
        used = [p[0]["rel"] for p in pairs]
        LOG.info("emparejado por CONTENIDO (miniaturas): %d emparejadas, %d sin emparejar %s; originales usados dos veces: %d", len(pairs), len(unmatched), unmatched[:5], len(used) - len(set(used)))
    if len(pairs) < 10:
        LOG.error("emparejadas insuficientes; revisa nombres en %s", a.vae_dir); sys.exit(2)
    F_o, F_v, dev, sizes = [], [], [], []
    for s, po, pv in pairs:
        fv, sv_ = feats_of(pv); fo, so_ = feats_of(po, sv_)
        F_o.append(fo); F_v.append(fv); dev.append(s["rel"] if a.orig_dir else s["rel"].split(os.sep)[0]); sizes.append((so_, sv_))
    LOG.info("dispositivos entre las %d: %d; tamano original/VAE (primeras 3): %s; VAE con tamano distinto del original: %d", len(pairs), len(set(dev)), sizes[:3], sum(o != v for o, v in sizes))
    ctl = read_jsonl(os.path.join(a.controls_dir, "h1-controls.jsonl"))
    ai = [r for r in ctl if r["label"] == "fake"]
    reals_train = [r for r in ctl if r["label"] == "real" and (a.orig_dir or r["device"] not in set(dev))]
    fam_ai = np.array([r["group"] for r in ai])
    LOG.info("IA: %d (%d familias); reales de entrenamiento (dispositivos distintos de los %d): %d", len(ai), len(set(fam_ai)), len(set(dev)), len(reals_train))
    result = {"n_pares": len(pairs), "n_dispositivos": len(set(dev))}
    for feat in FEATS:
        Xo = np.array([x[feat] for x in F_o]); Xv = np.array([x[feat] for x in F_v])
        X_ai = np.array([r["arms"]["mod8"][feat] for r in ai]); X_tr = np.array([r["arms"]["mod8"][feat] for r in reals_train])
        auc_a, ci_a = scenario_A(Xo, Xv, dev, a.n_boot)
        bc = scenario_BC(Xo, Xv, dev, X_ai, fam_ai, X_tr, a.n_boot)
        result[feat] = {"A_VAE_vs_original": {"auc": auc_a, "ic95": ci_a}, **{k: {"auc": v[0], "ic95": v[1], "por_familia": v[2]} for k, v in bc.items()}}
        if a.orig_dir:
            bd = scenario_BC(Xo, Xv, dev, X_ai, fam_ai, None, a.n_boot, indomain=True)
            result[feat].update({k + "_en_dominio": {"auc": v[0], "ic95": v[1], "por_familia": v[2]} for k, v in bd.items()})
    LOG.info("| escenario | feature | AUC | IC95 |")
    rows = [("A_VAE_vs_original", "A) VAE vs Original"), ("B_IA_vs_VAE", "B) IA vs VAE"), ("C_IA_vs_original", "C) IA vs Original")]
    if a.orig_dir:
        rows += [("B_IA_vs_VAE_en_dominio", "B) IA vs VAE [en dom]"), ("C_IA_vs_original_en_dominio", "C) IA vs Orig [en dom]")]
    for name, label in rows:
        for feat in FEATS:
            r = result[feat][name]
            LOG.info("| %-22s | %-8s | %.3f | [%.3f, %.3f] |", label, feat, r["auc"], *r["ic95"])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        LOG.info("escrito %s", a.out)
    return result


def self_test():
    import shutil
    from PIL import ImageFilter
    rng = np.random.default_rng(6)
    tmp = tempfile.mkdtemp()
    try:
        yy, xx = np.mgrid[0:700, 0:800]
        base = 0.5 + 0.2 * np.sin(xx / 60.0) * np.cos(yy / 50.0)
        real_dir, vae_dir, run_dir, cdir = (os.path.join(tmp, n) for n in ("reals", "vae", "run", "ctl"))
        for d in (vae_dir, run_dir, cdir):
            os.makedirs(d)
        sample, ctl = [], []
        for d in range(12):
            os.makedirs(os.path.join(real_dir, f"dev{d}"))
            for i in range(5):
                tilt = 0.3 * np.sin(xx / rng.uniform(40, 200) + rng.uniform(0, 6)) * np.cos(yy / rng.uniform(40, 200) + rng.uniform(0, 6))
                im = Image.fromarray((np.clip(base[..., None] + tilt[..., None] + rng.normal(0, 0.03, (700, 800, 3)), 0, 1) * 255).astype(np.uint8))
                im.save(os.path.join(real_dir, f"dev{d}", f"x{d}_{i}.jpg"), quality=92)
                sample.append({"key": f"real/dev{d}/x{d}_{i}.jpg", "group": "real", "label": "real", "rel": f"dev{d}/x{d}_{i}.jpg"})
                feats = CTL.features(CTL.aligned_crop(im))
                ctl.append({"key": sample[-1]["key"], "group": "real", "label": "real", "device": f"dev{d}", "status": "ok", "arms": {"mod8": {"lf_eq": feats[0], "res_spec": feats[1]}}})
                if d < 5:  # nombres sin relacion con el original; recorte central 512 como la VAE real
                    center_crop(im, (512, 512)).filter(ImageFilter.GaussianBlur(1.5)).save(os.path.join(vae_dir, f"real_vae_{d * 5 + i:03d}.png"))
        for fam in ("a", "b", "c"):
            for i in range(20):
                im = Image.fromarray((np.clip(base[:512, :512, None] + rng.normal(0, 0.005, (512, 512, 3)), 0, 1) * 255).astype(np.uint8))
                feats = CTL.features(im)
                ctl.append({"key": f"{fam}{i}", "group": fam, "label": "fake", "device": None, "status": "ok", "arms": {"mod8": {"lf_eq": feats[0], "res_spec": feats[1]}}})
                sample.append({"key": f"{fam}{i}", "group": fam, "label": "fake", "rel": f"{fam}/{i}.png"})
        with open(os.path.join(run_dir, "sample.json"), "w") as f:
            json.dump(sample, f)
        with open(os.path.join(cdir, "h1-controls.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in ctl))
        a = argparse.Namespace(orig_dir=None, run_dir=run_dir, real_dir=real_dir, vae_dir=vae_dir, controls_dir=cdir, out=None, n_boot=30)
        r = run(a)
        assert r["n_pares"] == 25 and r["n_dispositivos"] == 5
        assert r["res_spec"]["A_VAE_vs_original"]["auc"] > 0.9
        assert 0 <= r["lf_eq"]["C_IA_vs_original"]["ic95"][0] <= r["lf_eq"]["C_IA_vs_original"]["ic95"][1] <= 1
        od = os.path.join(tmp, "orig512"); os.makedirs(od)
        for j in range(25):  # originales 512 en orden: la i-esima corresponde a real_vae_i
            d_, i_ = divmod(j, 5)
            center_crop(Image.open(os.path.join(real_dir, f"dev{d_}", f"x{d_}_{i_}.jpg")), (512, 512)).save(os.path.join(od, f"o{j:03d}.png"))
        a2 = argparse.Namespace(run_dir=None, real_dir=None, orig_dir=od, vae_dir=vae_dir, controls_dir=cdir, out=None, n_boot=30)
        r2 = run(a2)
        assert r2["n_pares"] == 25 and r2["res_spec"]["A_VAE_vs_original"]["auc"] > 0.9 and "C_IA_vs_original_en_dominio" in r2["lf_eq"]
        LOG.info("autotest OK (5 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--run-dir"); p.add_argument("--real-dir"); p.add_argument("--orig-dir"); p.add_argument("--vae-dir"); p.add_argument("--controls-dir"); p.add_argument("--out")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("vae_dir", "controls_dir") + (() if a.orig_dir else ("run_dir", "real_dir")):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    run(a)


if __name__ == "__main__":
    main()
