#!/usr/bin/env python3
"""H1-L2: generacion de IA propia con historial CONOCIDO (PNG nativo 512x512, sin reescalar ni JPEG). Ver PREREGISTRO_H1.md seccion 19.

Generadores: sdturbo (stabilityai/sd-turbo, 2 pasos, guidance 0; primario) y sd15 (stable-diffusion-v1-5, 20 pasos, guidance 7.5; secundario,
misma familia que sd1 de Synthbuster). Prompts y semillas fijos en este fichero. Guarda PNG + owngen-manifest.jsonl (prompt, semilla, modelo,
SHA-256 del PNG, versiones). Reanudable. Necesita torch y diffusers (CPU).
Uso:
  python3 h1-owngen-make.py --out DIR [--n-sdturbo 100] [--n-sd15 40] [--only sdturbo|sd15]
  python3 h1-owngen-make.py --self-test
"""
import argparse, hashlib, json, logging, os, sys, tempfile
import numpy as np

LOG = logging.getLogger("h1-owngen-make")
SEED0 = 20261112
SCENES = ["a street market at noon", "a kitchen table with fruit", "a mountain lake in autumn", "a dog on a sofa", "a city bus stop in rain",
          "a bicycle leaning on a brick wall", "a bowl of soup on a wooden table", "a child playing in a park", "a train station platform",
          "a laptop on a desk near a window", "a harbor with fishing boats", "a living room with a plant", "a cyclist on a country road",
          "a cup of coffee on a cafe table", "a snowy village street", "a farmer in a field", "a beach with umbrellas", "an old church square",
          "a group of friends at a picnic", "a parked car on a suburban street"]
SCENES2 = ["a bakery counter with bread", "a quiet alley at dusk", "a market stall with vegetables", "a runner on a forest path", "a bus interior with passengers",
           "a garden with a wooden bench", "a rainy window with a cup of tea", "a construction site in the morning", "a library reading room", "a ferry crossing a river",
           "a classroom with students", "a rooftop with laundry", "a campfire by a lake", "a motorbike on a dirt road", "a supermarket aisle",
           "a pier at sunset", "a barber shop interior", "a playground in winter", "a vineyard in summer", "a tram on a cobbled street"]
STYLES = ["amateur photo taken with a smartphone", "candid photograph, natural light"]
GENS = {
    "sdturbo": {"model": "stabilityai/sd-turbo", "steps": 2, "guidance": 0.0},
    "sd15": {"model": "stable-diffusion-v1-5/stable-diffusion-v1-5", "steps": 20, "guidance": 7.5},
}


def prompt_of(i, new=False):
    sc = SCENES2 if new else SCENES
    return f"{STYLES[(i // len(sc)) % len(STYLES)]}, {sc[i % len(sc)]}"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pipe(name):
    import torch
    from diffusers import AutoPipelineForText2Image
    kw = {"torch_dtype": torch.float32, "low_cpu_mem_usage": True}
    if name == "sd15":
        kw["safety_checker"] = None
    pipe = AutoPipelineForText2Image.from_pretrained(GENS[name]["model"], **kw).to("cpu")
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def versions():
    out = {}
    for m in ("torch", "diffusers", "transformers"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def make(a, pipe_factory=load_pipe):
    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "owngen-manifest.jsonl")
    done = set()
    if os.path.exists(man_path):
        done = {json.loads(l)["file"] for l in open(man_path, encoding="utf-8") if l.strip()}
    plan = [(n, k) for n, k in (("sdturbo", a.n_sdturbo), ("sd15", a.n_sd15)) if a.only in (None, n) and k > 0]
    for name, count in plan:
        todo = [i for i in range(count) if f"{name}_{i:03d}.png" not in done]
        LOG.info("%s: %d por generar (%d ya hechas)", name, len(todo), count - len(todo))
        if not todo:
            continue
        pipe = pipe_factory(name); g = dict(GENS[name])
        if getattr(a, "steps", None):
            g["steps"] = a.steps
        off, new = getattr(a, "seed_offset", 0), getattr(a, "new_prompts", False)
        for i in todo:
            seed = SEED0 + (0 if name == "sdturbo" else 100000) + off + i
            img = pipe(prompt=prompt_of(i, new), num_inference_steps=g["steps"], guidance_scale=g["guidance"], height=512, width=512, generator=__import__("torch").Generator("cpu").manual_seed(seed)).images[0] if pipe_factory is load_pipe else pipe(prompt_of(i, new), seed)
            if img.size != (512, 512):
                raise ValueError(f"tamano inesperado {img.size}")
            fn = f"{name}_{i:03d}.png"; path = os.path.join(a.out, fn)
            img.convert("RGB").save(path, format="PNG")
            rec = {"file": fn, "generator": name, "prompt": prompt_of(i, new), "seed": seed, "sha256": sha256_file(path), **g, "dtype": "float32", "versions": versions()}
            with open(man_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if (i + 1) % 10 == 0:
                LOG.info("%s %d/%d", name, i + 1, count)
            if i == todo[0]:
                LOG.info("primera imagen %s generada", fn)
    LOG.info("fin; manifiesto %s", man_path)
    return man_path


def self_test():
    import shutil
    from PIL import Image
    tmp = tempfile.mkdtemp()
    try:
        class Fake:
            def __call__(self, prompt, seed):
                class R: pass
                r = R(); r.images = [Image.fromarray(np.random.default_rng(seed).integers(0, 255, (512, 512, 3), dtype=np.uint8))]
                return r
        # el pipe falso devuelve un objeto con .images; adaptamos a la firma usada con el pipe de diffusers
        class Pipe:
            def __call__(self, prompt, seed): return Fake()(prompt, seed).images[0]
        a = argparse.Namespace(out=tmp, n_sdturbo=3, n_sd15=2, only=None)
        p = make(a, lambda name: Pipe())
        rows = [json.loads(l) for l in open(p)]
        assert len(rows) == 5 and len({r["seed"] for r in rows}) == 5, "recuento/semillas"
        assert all(sha256_file(os.path.join(tmp, r["file"])) == r["sha256"] for r in rows), "hash"
        assert make(a, lambda name: Pipe()) == p and len(open(p).readlines()) == 5, "reanudacion"
        assert len({prompt_of(i) for i in range(40)}) == 40 and all(Image.open(os.path.join(tmp, r["file"])).size == (512, 512) for r in rows), "prompts/tamano"
        # etapa 1 de H2: semillas y prompts disjuntos de los originales, pasos configurables, mismos prompts/semillas entre condiciones
        orig_p = {prompt_of(i) for i in range(100)}; new_p = {prompt_of(i, True) for i in range(30)}
        assert not orig_p & new_p and len(new_p) == 30, "prompts nuevos disjuntos"
        out = {}
        for st in (1, 2, 4):
            d = os.path.join(tmp, f"s{st}")
            m = make(argparse.Namespace(out=d, n_sdturbo=3, n_sd15=0, only="sdturbo", steps=st, seed_offset=200000, new_prompts=True), lambda name: Pipe())
            out[st] = [json.loads(l) for l in open(m)]
        assert all(r["steps"] == st for st, rs in out.items() for r in rs), "pasos"
        assert all(r["seed"] >= SEED0 + 200000 for rs in out.values() for r in rs), "semillas fuera del rango original"
        assert [r["seed"] for r in out[1]] == [r["seed"] for r in out[4]] and [r["prompt"] for r in out[1]] == [r["prompt"] for r in out[2]], "emparejado entre condiciones"
        LOG.info("autotest OK (7 comprobaciones)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true"); p.add_argument("--out")
    p.add_argument("--n-sdturbo", type=int, default=100); p.add_argument("--n-sd15", type=int, default=40); p.add_argument("--only", choices=("sdturbo", "sd15"))
    p.add_argument("--steps", type=int, help="sustituye los pasos del generador (H2 etapa 1)")
    p.add_argument("--seed-offset", type=int, default=0, help="se suma a la semilla (H2: 200000)")
    p.add_argument("--new-prompts", action="store_true", help="usa SCENES2, disjunta de los 100 prompts originales")
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if not a.out:
        p.error("falta --out")
    make(a)


if __name__ == "__main__":
    main()
