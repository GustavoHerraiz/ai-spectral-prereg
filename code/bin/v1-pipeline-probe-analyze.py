#!/usr/bin/env python3
"""Análisis de la sonda de procesado de Detector v1.

Protocolo: l1-manisha2025/PREREGISTRO_V1_PIPELINE_PROBE.md (las reglas de decisión de este archivo
son las de la sección 8 de ese documento y no se ajustan tras ver datos).

Entrada : --run-dir con features.shard*.jsonl, sample.json y meta.json (salida de v1-pipeline-probe.mjs)
Salida  : analysis.json en --run-dir y un resumen en el log.

Uso:
    python3 l1-manisha2025/bin/v1-pipeline-probe-analyze.py --self-test
    python3 l1-manisha2025/bin/v1-pipeline-probe-analyze.py --run-dir l1-manisha2025/results/v1-pipeline-probe-run1
"""
import argparse
import glob
import hashlib
import json
import logging
import os
import sys
import tempfile

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("v1probe-analysis")
FEATURES = ("jpegPhaseRange", "phaseCoherence")
ARMS = ("C0", "C1", "R0", "R1")
SEED = 20261111
N_BOOT = 2000
MIN_FAMILY_PAIRED = 30
# Reglas de decisión (sección 8 del preregistro)
PIPE_MATCH_MAX, PIPE_MIS_MIN, PIPE_ONLY_MIN = 0.60, 0.70, 0.65
RESID_MIN, RESID_CI_LOW_MIN = 0.65, 0.55


def auc(pos, neg):
    """AUC con IA como clase positiva (dirección 'IA-alta', fijada por v1)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ranks = rankdata(np.concatenate([pos, neg]))
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def oriented(value):
    return max(value, 1 - value)


def load(run_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "features.shard*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    ok = [r for r in records if r.get("status") == "ok"]
    LOG.info("registros: %d ok, %d con error", len(ok), len(records) - len(ok))
    unique, n_dup = {}, 0
    for record in ok:
        prev = unique.get(record["key"])
        if prev is None:
            unique[record["key"]] = record
        elif prev == record:
            n_dup += 1
        else:
            raise ValueError(f"clave duplicada con contenido distinto {record['key']}")
    if n_dup:
        LOG.warning("%d registros duplicados identicos descartados (reanudacion/carrera de arranque)", n_dup)
    ok = list(unique.values())
    LOG.info("registros unicos: %d", len(ok))
    return ok, len(records) - len(ok)


def split(records):
    """Conjunto pareado: imágenes con las 4 variantes. Familias con >= MIN_FAMILY_PAIRED."""
    paired = [r for r in records if all(a in r["arms"] for a in ARMS)]
    reals = [r for r in paired if r["label"] == "real"]
    families = {}
    for record in paired:
        if record["label"] == "fake":
            families.setdefault(record["group"], []).append(record)
    families = {k: v for k, v in sorted(families.items()) if len(v) >= MIN_FAMILY_PAIRED}
    return reals, families


def values(records, arm, k):
    return np.array([r["arms"][arm]["f"][k] for r in records], float)


def macro_auc(real_arm, fake_arms, ri, fi):
    """AUC macro por familia; ri/fi son índices (re)muestreados de reales y de cada familia."""
    return float(np.mean([auc(fake_arms[f][fi[f]], real_arm[ri]) for f in fake_arms]))


def bootstrap(fn, n_real, n_fam, rng, n_boot=N_BOOT):
    out = []
    for _ in range(n_boot):
        ri = rng.integers(0, n_real, n_real)
        fi = {f: rng.integers(0, n, n) for f, n in n_fam.items()}
        out.append(fn(ri, fi))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def decide(arm_auc, arm_ci_low, a_mis, a_pipe_r1):
    """Reglas fijadas de antemano (preregistro sección 8)."""
    mean_match = float(np.mean(list(arm_auc.values())))
    if mean_match <= PIPE_MATCH_MAX and a_mis >= PIPE_MIS_MIN and a_pipe_r1 >= PIPE_ONLY_MIN:
        return "EXPLICADO_POR_PROCESADO"
    if all(v >= RESID_MIN for v in arm_auc.values()) and all(v > RESID_CI_LOW_MIN for v in arm_ci_low.values()):
        return "SENAL_RESIDUAL"
    return "MIXTO"


def paired_effect(delta, rng, n_boot=N_BOOT):
    delta = np.asarray(delta, float)
    meds = [np.median(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(n_boot)]
    p = wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0
    sd = delta.std(ddof=1)
    return {"n": int(len(delta)), "median_delta": float(np.median(delta)),
            "ci95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "dz": float(delta.mean() / sd) if sd > 0 else None, "wilcoxon_p": float(p)}


def analyze_feature(k, reals, families, rng):
    n_real = len(reals)
    n_fam = {f: len(v) for f, v in families.items()}
    real_arm = {a: values(reals, a, k) for a in ARMS}
    fake_arm = {a: {f: values(v, a, k) for f, v in families.items()} for a in ARMS}
    result = {"familias": list(families), "n_real_pareadas": n_real, "n_por_familia": n_fam}
    arm_auc, arm_ci = {}, {}
    for arm in ARMS:
        point = macro_auc(real_arm[arm], fake_arm[arm], np.arange(n_real), {f: np.arange(n) for f, n in n_fam.items()})
        ci = bootstrap(lambda ri, fi, arm=arm: macro_auc(real_arm[arm], fake_arm[arm], ri, fi), n_real, n_fam, rng)
        arm_auc[arm], arm_ci[arm] = point, ci
    result["auc_emparejada"] = {a: {"auc_macro": arm_auc[a], "ci95": arm_ci[a],
                                      "por_familia": {f: float(auc(fake_arm[a][f], real_arm[a])) for f in families}} for a in ARMS}
    ident = {f: np.arange(n) for f, n in n_fam.items()}
    mis_point = macro_auc(real_arm["R1"], fake_arm["C0"], np.arange(n_real), ident)
    mis_ci = bootstrap(lambda ri, fi: macro_auc(real_arm["R1"], fake_arm["C0"], ri, fi), n_real, n_fam, rng)
    result["auc_desemparejada_real_R1_vs_IA_C0"] = {"auc_macro": mis_point, "ci95": mis_ci}
    pipe = {}
    for arm in ("C1", "R0", "R1"):
        raw = float(auc(real_arm[arm], real_arm["C0"]))
        pipe[arm] = {"auc_vs_C0": raw, "auc_orientada": oriented(raw), "signo": "sube" if raw > 0.5 else "baja",
                     **paired_effect(real_arm[arm] - real_arm["C0"], rng)}
    result["efecto_procesado_en_reales"] = pipe
    fake_delta = {}
    for arm in ("C1", "R0", "R1"):
        delta = np.concatenate([fake_arm[arm][f] - fake_arm["C0"][f] for f in families])
        eff = paired_effect(delta, rng)
        real_delta = real_arm[arm] - real_arm["C0"]
        eff["mediana_reales_menos_ia"] = float(np.median(real_delta) - np.median(delta))
        eff["mannwhitney_p_reales_vs_ia"] = float(mannwhitneyu(real_delta, delta).pvalue)
        fake_delta[arm] = eff
    result["efecto_procesado_en_ia"] = fake_delta
    pooled = np.concatenate([fake_arm["C0"][f] for f in families])
    sd = np.sqrt((pooled.var(ddof=1) + real_arm["C0"].var(ddof=1)) / 2)
    result["d_cohen_IA_vs_real_C0"] = float((pooled.mean() - real_arm["C0"].mean()) / sd)
    result["veredicto"] = decide(arm_auc, {a: arm_ci[a][0] for a in ARMS}, mis_point, pipe["R1"]["auc_orientada"])
    return result


def secondary_2d(reals, families):
    """Exploratorio: regresión logística con las 2 features, AUC fuera de pliegue."""
    def oof(real_arm, fake_arm):
        fakes = [r for f in families.values() for r in f]
        x = np.vstack([[r["arms"][fake_arm]["f"] for r in fakes], [r["arms"][real_arm]["f"] for r in reals]])
        y = np.r_[np.ones(len(fakes)), np.zeros(len(reals))]
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
        p = cross_val_predict(model, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        return float(auc(p[y == 1], p[y == 0]))
    return {"emparejada_C0": oof("C0", "C0"), "emparejada_R1": oof("R1", "R1"), "desemparejada_realR1_vs_iaC0": oof("R1", "C0")}


def diagnostics(records):
    """Sesgo de formato de los datos de partida (no de las variantes): bytes y tamaño del archivo fuente."""
    reals = [r for r in records if r["label"] == "real"]
    out = {}
    for fam in sorted({r["group"] for r in records if r["label"] == "fake"}):
        fakes = [r for r in records if r["group"] == fam]
        out[fam] = {"n": len(fakes), "auc_bytes_fuente_vs_real": float(auc([r["bytes"] for r in fakes], [r["bytes"] for r in reals])),
                    "tamanos_modales": sorted({f"{r['width']}x{r['height']}" for r in fakes})[:4],
                    "n_con_R": int(sum("R0" in r["arms"] for r in fakes))}
    return out


def run(run_dir):
    records, n_err = load(run_dir)
    if not records:
        raise SystemExit("sin registros válidos")
    reals, families = split(records)
    if len(reals) < MIN_FAMILY_PAIRED or not families:
        raise SystemExit(f"conjunto pareado insuficiente: reales={len(reals)}, familias={list(families)}")
    LOG.info("pareadas: %d reales; familias %s", len(reals), {f: len(v) for f, v in families.items()})
    rng = np.random.default_rng(SEED)
    out = {"protocolo": "PREREGISTRO_V1_PIPELINE_PROBE.md", "n_errores_extraccion": n_err, "bootstrap": N_BOOT, "semilla": SEED,
           "estado": "exploratorio hasta congelar el preregistro", "diagnostico_formato_fuente": diagnostics(records),
           "features": {}, "secundario_2d_exploratorio": secondary_2d(reals, families)}
    if os.path.exists(os.path.join(run_dir, "meta.json")):
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as handle:
            out["meta"] = {k: v for k, v in json.load(handle).items() if k != "inventory"}
    for k, name in enumerate(FEATURES):
        out["features"][name] = analyze_feature(k, reals, families, rng)
        res = out["features"][name]
        LOG.info("== %s: %s", name, res["veredicto"])
        for arm in ARMS:
            a = res["auc_emparejada"][arm]
            LOG.info("   AUC emparejada %s = %.3f [%.3f, %.3f]", arm, a["auc_macro"], *a["ci95"])
        m = res["auc_desemparejada_real_R1_vs_IA_C0"]
        LOG.info("   AUC desemparejada (real R1 vs IA C0) = %.3f [%.3f, %.3f]", m["auc_macro"], *m["ci95"])
        for arm, e in res["efecto_procesado_en_reales"].items():
            LOG.info("   efecto %s en reales: AUC %.3f (%s), mediana d=%.4f dz=%s", arm, e["auc_vs_C0"], e["signo"], e["median_delta"], e["dz"])
        LOG.info("   d de Cohen IA vs real en C0 = %.2f", res["d_cohen_IA_vs_real_C0"])
    with open(os.path.join(run_dir, "analysis.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    LOG.info("2D exploratorio: %s", out["secundario_2d_exploratorio"])
    return out


def synthetic(scenario, seed=1, n_real=120, n_fam=60, families=("a", "b", "c")):
    """Corpus sintético con efectos conocidos. Escenarios: pipeline, residual, nulo."""
    rng = np.random.default_rng(seed)
    pipeline = {"C0": 0.0, "C1": -0.6, "R0": -0.6, "R1": -1.3} if scenario == "pipeline" else {"C0": 0.0, "C1": -0.3, "R0": -0.3, "R1": -0.6}
    class_effect = 1.4 if scenario == "residual" else 0.0
    shift = pipeline if scenario != "nulo" else {a: 0.0 for a in ARMS}
    rows = []
    for label, group, n in [("real", "real", n_real)] + [("fake", f, n_fam) for f in families]:
        for i in range(n):
            base = rng.normal(0, 1, 2) + (class_effect if label == "fake" else 0)
            arms = {a: {"f": list(base + shift[a] + rng.normal(0, 0.3, 2)), "sha256": "x"} for a in ARMS}
            rows.append({"key": f"{group}/{i}", "group": group, "label": label, "status": "ok", "arms": arms, "bytes": 1000 + i,
                         "width": 1024, "height": 1024})
    return rows


def self_test():
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        LOG.info("[%s] %s", "OK" if cond else "FALLO", name)

    rng = np.random.default_rng(0)
    x, y = rng.normal(0.5, 1, 40), rng.normal(0, 1, 50)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in x for b in y])
    check("T1 AUC por rangos == fuerza bruta", abs(auc(x, y) - brute) < 1e-12)
    for scenario, expected in [("pipeline", "EXPLICADO_POR_PROCESADO"), ("residual", "SENAL_RESIDUAL"), ("nulo", "MIXTO")]:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "features.shard0of1.jsonl"), "w", encoding="utf-8") as handle:
                for row in synthetic(scenario):
                    row["resizeEligible"] = True
                    handle.write(json.dumps(row) + "\n")
            global N_BOOT
            saved, N_BOOT = N_BOOT, 200
            try:
                out = run(tmp)
            finally:
                N_BOOT = saved
            verdicts = {name: out["features"][name]["veredicto"] for name in FEATURES}
            check(f"T2 escenario '{scenario}' -> {expected}", all(v == expected for v in verdicts.values()))
    check("T3 familias con <30 pareadas se excluyen", "x" not in split(synthetic("nulo", families=("a",)) + [dict(r, group="x", key="x/" + r["key"]) for r in synthetic("nulo", n_real=0, n_fam=5, families=("x",))])[1])
    check("T4 duplicado identico se descarta, distinto se rechaza", _dup_rejected())
    LOG.info("autotest: %d/%d correctos", sum(ok), len(ok))
    return all(ok)


def _dup_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        row = synthetic("nulo", n_real=1, n_fam=1, families=("a",))[0]
        with open(os.path.join(tmp, "features.shard0of1.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        identical_ok = len(load(tmp)[0]) == 1
        with open(os.path.join(tmp, "features.shard0of1.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n" + json.dumps(dict(row, label="x")) + "\n")
        try:
            load(tmp)
        except ValueError:
            return identical_ok
    return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if self_test() else 1
    if not args.run_dir:
        parser.error("indica --run-dir o --self-test")
    run(args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
