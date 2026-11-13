#!/usr/bin/env python3
"""H1-L2: generalizacion entre corpus de reales (LOCO). Ver PREREGISTRO_H1.md seccion 11.

Corpus de reales: coco (50, ya 512x512 PNG), run3 (205 nativos, 41 dispositivos), dresden (<=10/camara, misma regla y semilla
que h1-extract.py). Clase positiva: 700 IA Synthbuster (7 familias, modelo LOFO como en h1-post.py).
Pipeline comun P: recorte cuadrado central (lado menor) -> LANCZOS a 512 (si no es 512) -> rasgos `res_spec`, `lf_eq` (mismas funciones).
Brazos:
  igual   reales e IA pasan por P (decisivo: mismo procesado en ambas clases)
  nativo  reales por P; IA como en h1-controls-run1 brazo mod8 (recorte 512 nativo, sin reescalar) (referencia)
Pruebas: (1) entrenar con un corpus de reales y evaluar en cada otro (6 pares); (2) entrenar con dos y evaluar en el tercero;
(3) referencia en dominio (pliegues por dispositivo); (4) identificacion del corpus solo con reales (exactitud balanceada).
Uso:
  python3 h1-loco.py --run-dir RUN3 --real-dir RUN3REALS --synth-root DIR --coco-dir DIR --dresden-glob 'GLOB' \
     --controls-dir h1-controls-run1 --out OUT [--workers 2]
  python3 h1-loco.py --self-test
"""
import argparse, glob, json, logging, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image

LOG = logging.getLogger("h1-loco")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


CTL = load_module("h1ctl", os.path.join(HERE, "h1-controls-extract.py"))
EX = CTL.EX
ANA = CTL.load_module("h1analyze", os.path.join(HERE, "h1-analyze.py"))
SEED = ANA.SEED
FEATS = ("res_spec", "lf_eq")
REAL = ("coco", "run3", "dresden")
EXT = (".png", ".jpg", ".jpeg")
# criterios fijados en PREREGISTRO_H1.md seccion 11
AGUANTA_MEAN, AGUANTA_LO, ARTEFACTO_MEAN, CORPUS_ID_MIN = 0.75, 0.60, 0.68, 0.80


def pipeline_p(img):
    w, h = img.size
    s = min(w, h)
    if s < 512:
        raise ValueError("menor que 512")
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img if s == 512 else img.resize((512, 512), Image.LANCZOS)


def process(item):
    path, meta = item
    try:
        with Image.open(path) as src:
            img = pipeline_p(src.convert("RGB"))
        lf, sp = CTL.features(img)
        return {**meta, "status": "ok", "lf_eq": lf, "res_spec": sp}
    except Exception as exc:
        return {**meta, "status": "error", "error": str(exc)[:200]}


def build_items(a):
    items = []
    with open(os.path.join(a.run_dir, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    for s in sample:
        if s["label"] == "real":
            items.append((os.path.join(a.real_dir, s["rel"]), {"key": "run3/" + s["rel"], "corpus": "run3", "label": "real", "group": "real", "device": s["rel"].split(os.sep)[0]}))
        else:
            items.append((os.path.join(a.synth_root, s["rel"]), {"key": "ai/" + s["rel"], "corpus": "ai", "label": "fake", "group": s["group"], "device": None}))
    for i, p in enumerate(sorted(os.path.join(d, f) for d, _, fs in os.walk(a.coco_dir) for f in fs if f.lower().endswith(EXT))):
        items.append((p, {"key": "coco/" + os.path.basename(p), "corpus": "coco", "label": "real", "group": "real", "device": f"coco{i:03d}"}))
    rng = np.random.default_rng(SEED)
    for d in sorted(glob.glob(a.dresden_glob)):
        files = sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(EXT))
        pick = sorted(rng.choice(len(files), min(EX.DRESDEN_MAX, len(files)), replace=False)) if files else []
        device = os.path.basename(os.path.dirname(d))
        for i in pick:
            items.append((files[i], {"key": f"dresden/{device}/{os.path.basename(files[i])}", "corpus": "dresden", "label": "real", "group": "real", "device": device}))
    keys = [m["key"] for _, m in items]
    assert len(keys) == len(set(keys)), "claves duplicadas"
    return items


def extract(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h1-loco-feats.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    items = [it for it in build_items(a) if it[1]["key"] not in done]
    LOG.info("%d imagenes por hacer (%d ya hechas)", len(items), len(done))
    n = err = 0
    with open(path, "a", encoding="utf-8") as out, Pool(a.workers) as pool:
        for rec in pool.imap_unordered(process, items, chunksize=4):
            out.write(json.dumps(rec) + "\n"); out.flush(); n += 1; err += rec["status"] != "ok"
            if n % 100 == 0:
                LOG.info("%d/%d (errores %d)", n, len(items), err)
    LOG.info("extraccion fin: %d nuevas, errores %d", n, err)
    return path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(l) for l in f if l.strip()) if r["status"] == "ok"]


def ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def fit_scores(X_ai_tr, X_re_tr, targets):
    m = ANA.model().fit(np.vstack([X_ai_tr, X_re_tr]), np.r_[np.ones(len(X_ai_tr)), np.zeros(len(X_re_tr))])
    return [m.decision_function(t) for t in targets]


def macro_auc(S_ai, s_re, fams):
    return float(np.mean([ANA.auc(S_ai[f], s_re) for f in fams]))


def boot(S_ai, s_re, dev_re, fams, n_boot, rng):
    ud = sorted(set(dev_re)); idx = {d: np.where(np.asarray(dev_re) == d)[0] for d in ud}
    vals = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[ud[i]] for i in rng.choice(len(ud), len(ud))])
        vals.append(np.mean([ANA.auc(rng.choice(S_ai[f], len(S_ai[f])), s_re[sel]) for f in fams]))
    return ci(vals)


def cross_corpus(X_ai, fam_ai, X_re, corp, dev, train_corpora, test_corpus, n_boot):
    fams = sorted(set(fam_ai))
    tr = np.isin(corp, train_corpora); te = corp == test_corpus
    S_ai, s_re = {}, np.zeros(te.sum())
    for f in fams:
        sa, sr = fit_scores(X_ai[fam_ai != f], X_re[tr], [X_ai[fam_ai == f], X_re[te]])
        S_ai[f] = sa; s_re += sr / len(fams)
    # un unico vector de reales (media sobre modelos LOFO) para el IC; el punto usa lo mismo
    point = macro_auc(S_ai, s_re, fams)
    return {"auc": point, "ic95": boot(S_ai, s_re, dev[te], fams, n_boot, np.random.default_rng(SEED)), "por_familia": {f: float(ANA.auc(S_ai[f], s_re)) for f in fams}, "n_reales_test": int(te.sum())}


def in_domain(X_ai, fam_ai, X_re, corp, dev, corpus, n_boot):
    fams = sorted(set(fam_ai)); te = np.where(corp == corpus)[0]
    fold = ANA.device_folds(dev[te], np.random.default_rng(SEED)); nf = min(ANA.N_DEV_FOLDS, len(set(dev[te])))
    fold = fold % nf
    S_ai = {f: [] for f in fams}; s_re = np.zeros(len(te))
    for f in fams:
        for g in range(nf):
            sa, sr = fit_scores(X_ai[fam_ai != f], X_re[te[fold != g]], [X_ai[fam_ai == f], X_re[te[fold == g]]])
            S_ai[f].append(sa); s_re[fold == g] += sr / len(fams)
    S_ai = {f: np.mean(v, axis=0) for f, v in S_ai.items()}
    return {"auc": macro_auc(S_ai, s_re, fams), "ic95": boot(S_ai, s_re, dev[te], fams, n_boot, np.random.default_rng(SEED))}


def corpus_id(X_re, corp, dev):
    from sklearn.metrics import balanced_accuracy_score
    fold = ANA.device_folds(dev, np.random.default_rng(SEED)); pred = np.empty(len(corp), dtype=object)
    for g in range(ANA.N_DEV_FOLDS):
        tr, te = fold != g, fold == g
        m = ANA.model().set_params(logisticregression__class_weight="balanced").fit(X_re[tr], corp[tr])
        pred[te] = m.predict(X_re[te])
    return float(balanced_accuracy_score(corp, pred))


def verdict(res):
    cc = [v for k, v in res["cruzado"].items()]
    mean = float(np.mean([c["auc"] for c in cc])); lo = min(c["ic95"][0] for c in cc)
    if mean >= AGUANTA_MEAN and lo >= AGUANTA_LO:
        return "aguanta", mean, lo
    if mean < ARTEFACTO_MEAN:
        return "artefacto de corpus", mean, lo
    return "indeterminada", mean, lo


def analyze(a, feats_path):
    rows = read_jsonl(feats_path)
    ai_igual = [r for r in rows if r["corpus"] == "ai"]
    reals = [r for r in rows if r["corpus"] in REAL]
    ctl = [r for r in read_jsonl(os.path.join(a.controls_dir, "h1-controls.jsonl")) if r["label"] == "fake"]
    LOG.info("reales por corpus: %s; IA igual: %d; IA nativo: %d", {c: sum(r["corpus"] == c for r in reals) for c in REAL}, len(ai_igual), len(ctl))
    corp = np.array([r["corpus"] for r in reals]); dev = np.array([f'{r["corpus"]}:{r["device"]}' for r in reals])
    out = {"n_reales": {c: int((corp == c).sum()) for c in REAL}}
    for arm, ai in (("igual", ai_igual), ("nativo", None)):
        out[arm] = {}
        fam_ai = np.array([r["group"] for r in (ai_igual if ai is not None else ctl)])
        for feat in FEATS:
            X_re = np.array([r[feat] for r in reals])
            X_ai = np.array([r[feat] for r in ai_igual]) if arm == "igual" else np.array([r["arms"]["mod8"][feat] for r in ctl])
            res = {"cruzado": {}, "loco": {}, "en_dominio": {}}
            for tr in REAL:
                for te in REAL:
                    if tr != te:
                        res["cruzado"][f"{tr}->{te}"] = cross_corpus(X_ai, fam_ai, X_re, corp, dev, [tr], te, a.n_boot)
            for te in REAL:
                res["loco"][f"resto->{te}"] = cross_corpus(X_ai, fam_ai, X_re, corp, dev, [c for c in REAL if c != te], te, a.n_boot)
                res["en_dominio"][te] = in_domain(X_ai, fam_ai, X_re, corp, dev, te, a.n_boot)
            res["veredicto"], res["media_cruzado"], res["min_ic_inf"] = verdict(res)
            if arm == "igual":
                res["identificacion_corpus_exactitud_balanceada"] = corpus_id(X_re, corp, dev)
            out[arm][feat] = res
    LOG.info("| brazo | feature | prueba | AUC | IC95 |")
    for arm in ("igual", "nativo"):
        for feat in FEATS:
            r = out[arm][feat]
            for grp in ("cruzado", "loco", "en_dominio"):
                for k, v in r[grp].items():
                    LOG.info("| %-6s | %-8s | %-9s %-16s | %.3f | [%.3f, %.3f] |", arm, feat, grp, k, v["auc"], *v["ic95"])
            LOG.info("RESUMEN %s %s: media cruzado %.3f, min IC inf %.3f -> %s", arm, feat, r["media_cruzado"], r["min_ic_inf"], r["veredicto"])
    for feat in FEATS:
        LOG.info("CORPUS-ID %s: exactitud balanceada %.3f (azar 0.333; umbral %.2f)", feat, out["igual"][feat]["identificacion_corpus_exactitud_balanceada"], CORPUS_ID_MIN)
    for feat in FEATS:
        LOG.info("VEREDICTO H1-L2-LOCO %s (brazo igual): %s", feat, out["igual"][feat]["veredicto"])
    with open(os.path.join(a.out, "h1-loco.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    LOG.info("escrito %s", os.path.join(a.out, "h1-loco.json"))
    return out


def self_test():
    import shutil
    rng = np.random.default_rng(11)
    tmp = tempfile.mkdtemp()
    try:
        yy, xx = np.mgrid[0:700, 0:800]
        def make(kind, size):
            tilt = 0.3 * np.sin(xx / rng.uniform(40, 200) + rng.uniform(0, 6)) * np.cos(yy / rng.uniform(40, 200) + rng.uniform(0, 6))
            arr = 0.5 + tilt[..., None] + rng.normal(0, 0.03, (700, 800, 3))
            if kind == "ai":
                arr += 0.04 * np.sin(xx * 2.4)[..., None]  # patron de alta frecuencia solo en IA
            return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).crop((0, 0, size[0], size[1]))
        run_dir, real_dir, synth, coco, dres, cdir, out = (os.path.join(tmp, n) for n in ("run", "r", "s", "c", "d", "ctl", "out"))
        for d in (run_dir, cdir, out, coco): os.makedirs(d)
        sample, ctl = [], []
        for d in range(8):
            os.makedirs(os.path.join(real_dir, f"dev{d}"))
            for i in range(3):
                make("real", (800, 700)).save(os.path.join(real_dir, f"dev{d}", f"x{i}.jpg"), quality=92)
                sample.append({"key": f"r{d}{i}", "group": "real", "label": "real", "rel": f"dev{d}/x{i}.jpg"})
        for i in range(12): make("real", (700, 700)).resize((512, 512), Image.LANCZOS).save(os.path.join(coco, f"c{i}.png"))
        for c in range(4):
            os.makedirs(os.path.join(dres, f"cam{c}", "nat"))
            for i in range(4): make("real", (780, 690)).save(os.path.join(dres, f"cam{c}", "nat", f"d{i}.jpg"), quality=90)
        for fam in ("a", "b", "c"):
            os.makedirs(os.path.join(synth, fam))
            for i in range(10):
                im = make("ai", (700, 600)); im.save(os.path.join(synth, fam, f"{i}.png"))
                sample.append({"key": f"{fam}{i}", "group": fam, "label": "fake", "rel": f"{fam}/{i}.png"})
                lf, sp = CTL.features(CTL.aligned_crop(im))
                ctl.append({"key": f"{fam}{i}", "group": fam, "label": "fake", "status": "ok", "arms": {"mod8": {"lf_eq": lf, "res_spec": sp}}})
        with open(os.path.join(run_dir, "sample.json"), "w") as f: json.dump(sample, f)
        with open(os.path.join(cdir, "h1-controls.jsonl"), "w") as f: f.write("\n".join(json.dumps(r) for r in ctl))
        a = argparse.Namespace(run_dir=run_dir, real_dir=real_dir, synth_root=synth, coco_dir=coco, dresden_glob=os.path.join(dres, "*", "nat"), controls_dir=cdir, out=out, workers=1, n_boot=20)
        p = extract(a)
        assert len(read_jsonl(p)) == 24 + 12 + 16 + 30, "recuento de extraccion"
        assert extract(a) == p and len(read_jsonl(p)) == 82, "reanudacion"
        # pipeline P: salida 512x512 y recorte cuadrado
        assert pipeline_p(make("real", (800, 700))).size == (512, 512)
        res = analyze(a, p)
        r = res["igual"]["res_spec"]
        assert all(v["auc"] > 0.8 for v in r["cruzado"].values()), "con huella inyectada el AUC cruzado debe ser alto"
        assert r["veredicto"] == "aguanta", r["veredicto"]
        # sin huella: IA = reales -> AUC ~ azar y veredicto de artefacto
        for rec_path in (p,):
            rows = [json.loads(l) for l in open(rec_path)]
        shuffled = [r_ for r_ in rows]
        real_pool = [r_ for r_ in rows if r_["corpus"] == "coco"]
        for i, r_ in enumerate(r_ for r_ in shuffled if r_["corpus"] == "ai"):
            r_["res_spec"], r_["lf_eq"] = real_pool[i % len(real_pool)]["res_spec"], real_pool[i % len(real_pool)]["lf_eq"]
        p2 = os.path.join(out, "null.jsonl")
        with open(p2, "w") as f: f.write("\n".join(json.dumps(x) for x in shuffled))
        null = analyze(a, p2)["igual"]["res_spec"]
        assert null["media_cruzado"] < 0.68 and null["veredicto"] == "artefacto de corpus", (null["media_cruzado"], null["veredicto"])
        LOG.info("autotest OK (6 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    for k in ("run-dir", "real-dir", "synth-root", "coco-dir", "dresden-glob", "controls-dir", "out"):
        p.add_argument("--" + k)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    for req in ("run_dir", "real_dir", "synth_root", "coco_dir", "dresden_glob", "controls_dir", "out"):
        if not getattr(a, req):
            p.error(f"falta --{req.replace('_', '-')}")
    analyze(a, extract(a))


if __name__ == "__main__":
    main()
