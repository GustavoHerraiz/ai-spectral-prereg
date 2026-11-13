"""Desglose descriptivo por familia de la sonda v1 (post hoc, exploratorio). AUC IA-alta con IC95 bootstrap (imagenes, B=2000).
Uso: python3 v1-pipeline-probe-by-family.py --run-dir DIR"""
import argparse, importlib.util, logging, os, sys
import numpy as np

LOG = logging.getLogger("by-family")
spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1-pipeline-probe-analyze.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", required=True); a = p.parse_args()
    records, _ = an.load(a.run_dir)
    reals, fams = an.split(records)
    rng = np.random.default_rng(20261111)
    LOG.info("n reales=%d; AUC IA-alta (0.5 = azar; <0.5 = reales mas altos); IC95 bootstrap", len(reals))
    for k, name in enumerate(an.FEATURES):
        for arm in ("C0", "R1"):
            r = np.array(an.values(reals, arm, k))
            for f, recs in fams.items():
                x = np.array(an.values(recs, arm, k))
                v = an.auc(x, r)
                bs = [an.auc(rng.choice(x, len(x)), rng.choice(r, len(r))) for _ in range(2000)]
                LOG.info("%-16s %s %-20s n=%d AUC=%.3f [%.3f, %.3f]", name, arm, f, len(x), v, *np.percentile(bs, [2.5, 97.5]))


if __name__ == "__main__":
    main()
