# The spectral separation between real photos and AI-generated images is explained by the processing chain, not by a generation fingerprint: a preregistered negative report

**Author:** Gustavo Herraiz. **Status:** preprint draft, pending external review. **Branch:** `main` (frozen base: tag `h1-congelada`).
**English translation of `PAPER.md` (Spanish, the reference version).** The translation was produced with an AI assistant and has not been reviewed by a native speaker; where they differ, `PAPER.md` prevails. Repository files (preregistrations, scripts, logs) are in Spanish.
**Full chronological record and all intermediate figures:** `MANUSCRITO.md` (Appendix A). Preregistrations: `PREREGISTRO_H1.md`, `PREREGISTRO_H1R.md`, `PREREGISTRO_H2.md`.

## Abstract

**Background.** Spectrum-based detectors of synthetic images are often evaluated on real and AI-generated images that differ in format, size and compression history, so part of their separation may be due to processing rather than origin. **Objective.** To determine how much of the real/AI separation achieved by two in-house spectral features (`res_spec`, `lf_eq`) survives when processing is controlled. **Methods.** Preregistered study (criteria and predictions fixed before seeing the data; self-tests with injected and null effects; everything post hoc labelled "exploratory") with 205 smartphone photos, 90 photos from Dresden cameras, the Synthbuster benchmark [4] (700 AI images, 7 families) and 100 images from an in-house generator with known history (sd-turbo, native 512×512). Validation is across real-image corpora (run3 ↔ Dresden) with 95% bootstrap CIs. **Results.** (i) With the same rescaling and format on both classes, separation drops from ≈0.8–0.9 to ≈0.5–0.6 AUC and does not generalize across corpora (`res_spec` 0.536; `lf_eq` 0.502). (ii) `res_spec` holds at native resolution (0.897) and after imitating the processing on the AI class only (0.856); with the in-house AI it reaches 0.942 if the chain (low-pass ×1.5 + JPEG q90 + q92) is applied to the AI class only. (iii) When **the same chain is applied to the real images**, `res_spec` falls to 0.635 (preregistered threshold 0.68; lower CI bound, minimum 0.544) and the model flags processed real images as "AI" (0.866). (iv) In an exploratory replication with 8 AI sets of unverified provenance, `res_spec` with matched chain gives 0.42–0.56 in all 8. **Conclusion.** In this benchmark and with these features, the observed separation reflects the processing chain and scale, not a generation fingerprint. Scope: in-house and proxy features, one benchmark and one confirmatory in-house generator; real images are downscaled ≥3× while the AI images are native at 512 (unresolved scale asymmetry). **Published detectors are not refuted.** Two open issues remain: a weak residual of `res_spec` (0.61–0.66) with sd-turbo and an unexplained discordance (§6.3). A later preregistered investigation (H2, §4.4) ruled out the SD lineage of the training set and sharpness as causes; the shift reproduces only with 2 sd-turbo steps and its origin remains unidentified.

**Keywords:** synthetic image detection, spectral fingerprint, preregistration, negative result, processing chain, PRNU.

## 1. Introduction

Detectors of generated images are usually validated with real and synthetic images of different provenance (format, size, compression). If the two classes differ in history, a classifier can learn the history. This work starts from replicating the ideas of Manisha et al. [1, 2] (L1 fingerprint) and Mallet et al. [3] (noise correlations between channels) and asks what part of the separation survives once processing is controlled. Two preregistered hypotheses are posed:

- **H1-PRNU:** the presence of the registry's sensor noise pattern is a signal of "real".
- **H1-L2:** a low/high-frequency fingerprint separates real images from AI images of unseen families after matching the processing.

The negative result is reported in the same detail a positive one would have received.

## 2. Methods

### 2.1 Data

| Corpus | n | Description |
|---|---|---|
| `run3` | 205 real, 41 devices (≤5 per device) | native smartphone photos, JPEG, shorter side ≥2000 px |
| Dresden `nat` | 90 real (≤10 per camera, seed 20261111) | camera JPEG |
| Synthbuster | 700 AI, 7 families | 100 per family; PNG except firefly (JPEG, 1792 px) |
| COCO-512 | 50 real | rescaled to 512×512, preparation undocumented |
| In-house AI (sd-turbo) | 100 | native 512×512 PNG, 2 steps, guidance 0, float32 with attention/VAE slicing (H1-D3); manifest with SHA-256 and versions |
| 8 `ai_*` sets | 50 each | provenance **unverified** (exploratory, §5.5) |

The v1 detector, the 81 K PRNU registry, v3 and the existing code are read-only.

### 2.2 Features and classifier

`res_spec`: radial log-power of the residual (3×3 median, grayscale), 64 logarithmic bins in [0.004, 0.5] cycles/px. `lf_eq`: L1-proxy vector of 32 bins after equalizing the processing (256 + JPEG Q90). Classifier: standardization + balanced L2 logistic regression. Metric: AUC (macro per family in LOFO; in cross-corpus validation, the mean of the two directions run3→Dresden and Dresden→run3). 95% CIs by bootstrap over devices (real) and images (AI), seed 20261111.

### 2.3 Processing chains

A0: unprocessed. A1: JPEG q92. A5: low-pass (Lanczos ×1.5 up and down) + JPEG q90 + q92. Real images: R0 (downscaling to shorter side 512, crop aligned to 8), R1 (R0 + q92), R5 (R0 with the A5 chain).

### 2.4 Preregistered design

Hypotheses, thresholds and predictions (with probabilities) were fixed and tagged in git before each run (`h1-criterios-s07…s21`, `h1r-criterios`). Self-tests with injected and null effects were run before each real run. Deviations were recorded before running (H1-D1 to D3). Verdict thresholds in cross-corpus validation: "holds" if mean ≥0.75 and lower CI bound minimum ≥0.60; "artifact" if mean <0.68; otherwise "indeterminate". This applies to all (c) comparisons in §4.

### 2.5 Chain analysis (§21)

(c) **primary, chain matched on both sides:** model trained with Synthbuster A5 and R5 of one corpus; tested with in-house AI A5 against R5 of the other. (a) diagnostic: A5/R1 model; AUC in-house AI A5 > in-house AI A1 (same images). (b) diagnostic: A5/R1 model; AUC R5 > R1 (same real images). High (a) and (b) indicate that the model detects the chain.

## 3. Results of the earlier stages

| Stage | Result | Reading |
|---|---|---|
| Probe v1 (run3) | `jpegPhaseRange` 0.612 [0.570, 0.655]; `phaseCoherence` 0.611 | MIXED; variation among real devices > real–AI difference |
| H1-PRNU (995 images) | real with K 0.931 [0.905, 0.954]; camera absent 0.587 [0.494, 0.675]; AI: firefly 0.56 above threshold, rest 0 | firefly's 56% is a generic JPEG artifact (53/100 won by a single K) |
| Initial H1-L2 (LOFO) | `lf_eq` 0.799 [0.776, 0.821]; `res_spec` 0.901 [0.885, 0.915] | byte baseline 0.993: format confound |
| Format controls | `res_spec` 0.774–0.916; `lf_eq` 0.581–0.877; real JPEG vs firefly (same format) `res_spec` 0.924 | partial verdict; prediction "`res_spec` collapses" refuted |
| VAE control (exploratory, 50 real) | VAE vs original 0.586 (`res_spec`); B ≈ C | the decoder (this VAE, at 512) does not explain the separation |
| LOCO, common 512 pipeline | `res_spec` 0.536; `lf_eq` 0.502 | corpus artifact (does not transfer) |
| Native LOCO (run3↔Dresden) | `res_spec` 0.897; `lf_eq` 0.733 | `res_spec` holds; prediction refuted |
| Scale dose-response | `res_spec` 0.897→0.781 with reduction 1→512; `lf_eq` 0.733→0.509 | `lf_eq` depends on scale; `res_spec` is not explained by scale |
| Resampling + double JPEG on AI only | `res_spec` 0.856 (lower CI bound minimum 0.797) | "persists" (**conclusion superseded by §4, see erratum in §7**) |

Details in `MANUSCRITO.md` §5.1–5.9.

## 4. Main results

### 4.1 In-house AI, chain on the AI class only (§20, n=100)

| Arm | `res_spec` (cross mean) | Verdict |
|---|---|---|
| A5/R1 (primary) | 0.942 [lower CI bound minimum 0.907] | confirmed (prediction p≈0.4 refuted) |
| A1/R1 | 0.509 | chance (Synthbuster: 0.781) |
| A0/R0 | 0.704 | indeterminate |

`lf_eq`: 0.885 in A5/R1 but 0.251 (A0/R0) and 0.219 (A1/R1), inverted: not a generation signal.

### 4.2 Chain control (§22, sd-turbo)

| Feature | Analysis | run3→Dresden | Dresden→run3 | Mean | Verdict |
|---|---|---|---|---|---|
| `res_spec` | **(c) matched chain (primary)** | 0.661 [0.586, 0.739] | 0.608 [0.544, 0.669] | **0.635** | **chain artifact** |
| `res_spec` | (a) AI A5 vs A1 | 0.640 [0.616, 0.673] | 0.643 [0.618, 0.678] | 0.642 | chain not detected |
| `res_spec` | (b) real R5 vs R1 | 0.904 [0.842, 0.962] | 0.827 [0.790, 0.866] | 0.866 | chain detected |
| `lf_eq` | (c) | 0.242 [0.163, 0.325] | 0.314 [0.257, 0.371] | 0.278 | chain artifact |

Predictions: (c) artifact p≈0.6, met; (b) detected p≈0.6, met; (a) detected p≈0.8, **refuted**.

### 4.3 Exploratory replication (H1-R, unverified provenance)

`res_spec` with matched chain (c), cross mean: Flux schnell 0.472; GPT-image 0.461; Kandinsky 2.2 0.525; Nano Banana 0.475; Nano Banana (generated, known prior JPEG and rescaling) 0.514; PixArt-Σ 0.421; SDXL 0.559; Würstchen 0.429. All 8 are "chain artifact". `lf_eq` (c): 0.433–0.497. A5 vs A1 (a): 0.845–0.964 in all 8. Predictions: majority <0.68, met (8/8); some "persists" (p≈0.3), did not occur; Nano Banana different from the rest (p≈0.5), refuted.

### 4.4 H2: investigation of the sd-turbo lead (branch `h2-sdturbo`, `PREREGISTRO_H2.md`)

Without a chain (A1), sd-turbo already scores as "AI" with the model trained on Synthbuster A5 vs R1 (AUC 0.899), which the 8 folders of §4.3 do not. Three preregistered analyses, with criteria and predictions fixed before running:

| stage | question | result | verdict |
|---|---|---|---|
| 0 | Stable Diffusion lineage in the training set? | without SD1+SD2+SDXL: 0.919 (lower CI bound minimum 0.867); controls 0.884–0.904 | not explained by SD lineage |
| 1 | does it reproduce (30 new images) and depend on the number of steps? | 2 steps 0.816 (0.704); 1 step 0.375; 4 steps 0.572; S(1)−S(4) = −0.197 [−0.311, −0.086] | reproduces; depends on steps, but **non-monotonic** |
| 2 | do sharpness or high frequency follow it? (exploratory) | Spearman 0.094 (`d_lap`) and 0.163 (`d_hf`), CIs include 0 | no descriptor |

The shift is real and reproducible, specific to the configuration (2 steps) and non-monotonic, which does not support a general distillation feature; **its cause was not identified**. Predictions: stage 0 (lineage, p≈0.35) not met; stage 1 (replication p≈0.75; steps p≈0.35) met; stage 2 (none, p≈0.30) met. A single model, n = 30 per condition. A `.tar` of the 90 images has SHA-256 `e49ffbb81504f233666563a0aad2bc7b857349b9df8885e3b370c4cde0ab53d8`.

## 5. Status of the evidence

- **Confirmatory (criteria fixed beforehand):** LOCO, native, scale, in-house AI (§4.1), chain control (§4.2).
- **Preregistered (criteria fixed beforehand), single-model scope:** H2 stages 0 and 1 (§4.4).
- **Exploratory:** VAE control, H1-R replication (§4.3), diagnostic of §6.3 (§5.13 of the manuscript), H2 stage 2 (§4.4).
- The 8 sets of §4.3 do not count as a confirmatory replication: no provenance, possible prior rescaling and no prompt control.

## 6. Discussion

### 6.1 What `res_spec` measures
The apparent `res_spec` fingerprint reproduces only when the chain is imposed on one class. With the chain on both sides it falls from 0.942 to 0.635, and processed real images score "AI" (0.866). This is consistent with the feature detecting processing (low-pass + double JPEG), not generation.

### 6.2 Scale and processing
`lf_eq` is a function of scale (0.733→0.509 with downscaling). `res_spec` decays less, but the real/AI asymmetry (real images downscaled ≥3×, AI native at 512) was never broken: there are no native ~512 real images. Until there are, any AUC of this design mixes scale with origin.

### 6.3 Open questions
1. **Weak residual.** (c) 0.61–0.66 with lower CI bound >0.5 in sd-turbo is not ruled out; it is below the preregistered threshold and does not reproduce in the 8 folders (0.42–0.56).
2. **Discordance.** In sd-turbo, A5 vs A1 is 0.64 but in the 8 folders it is 0.85–0.96. Post hoc analysis: sd-turbo already scores "AI" without the chain (A1: +1.5 to +2.6, versus −2.7 to −0.6 in the others) and the chain adds only ≈+1 (≈+3 in the others). The earlier hypothesis of "dispersion due to content" is **refuted**: the IQR of sd-turbo (2.1–2.3) is similar to the others (1.3–2.2); the difference is one of position. **H2 stage 0 (preregistered, `PREREGISTRO_H2.md` §7):** retraining without the SD1, SD2 and SDXL families does not reduce the shift (A1 vs R1: 0.899 → 0.919; controls 0.884–0.904), so the Stable Diffusion lineage of the training set does **not** explain it. **Stage 1 (`PREREGISTRO_H2.md` §9):** with 30 new images per condition, the shift reproduces with 2 steps (0.816; lower CI bound minimum 0.704) but is **non-monotonic in the number of steps** (1 step 0.375; 4 steps 0.572), so it does not support a distillation feature and is specific to the configuration; a single model, n = 30. **Stage 2 (`PREREGISTRO_H2.md` §11):** neither sharpness (`d_lap`, rho 0.09) nor the high-frequency fraction (`d_hf`, rho 0.16) follows the score; the texture hypothesis is refuted and **the cause remains unidentified**. Causes that cannot be distinguished: a generation property (2 steps, distilled), the in-house pipeline (float32, slicing) or histories of the other folders that erase the feature. This is a lead that requires preregistration, not a finding.

### 6.4 Relation to the literature
The results do not contradict [1, 3]: they evaluate in-house or proxy features on a different benchmark and training set. They do indicate that, in this design, controlling for processing removes the separation.

## 7. Conclusions

1. With `res_spec` and `lf_eq`, the real/AI separation in this benchmark **is explained by the processing chain and scale**, not by a generation fingerprint.
2. The decisive test: with the chain matched on both sides, `res_spec` falls to 0.635 (<0.68), and processed real images score "AI" (0.866).
3. The pattern is not contradicted by 8 additional sets of unverified provenance (0.42–0.56), on an exploratory basis.
4. H1-PRNU and H1-L2 are **not supported**.
5. **Erratum.** The conclusion of §5.9 ("persists" at 0.856) applied the chain to the AI class only and did not match the processing; it is superseded by §4.2.
6. **H2.** The "AI" score of sd-turbo without a chain is not explained by the SD lineage of the training set nor by sharpness or high frequency; it reproduces only with 2 steps and is non-monotonic in the number of steps. It is a property of that configuration, not a general "AI" fingerprint, and its cause remains **unidentified** (a single model).
7. For spectral detectors, the methodological lesson is: match the processing chain on both classes and verify that the model does not flag processed real images as AI.

## 8. Limitations

- One benchmark (Synthbuster, firefly in JPEG and at 1792 px) and one confirmatory in-house generator (sd-turbo, n=100, float32 with slicing); the 8 additional folders have unverified provenance.
- Scale asymmetry unbroken; real images with ≤5 (run3) or ≤10 (Dresden) images per device; only 3 real corpora; COCO-512 with an undocumented pipeline.
- Per-image bootstrap does not model dependence between images of the same device or scene in COCO and VAE.
- One SD VAE at 512; it does not represent DALL·E, Midjourney or Firefly.
- Results refer to these features and this training; they do not refute published detectors.
- Preregistrations are tagged in git but not cryptographically signed.
- A weak residual (0.61–0.66) is not ruled out.
- H2: a single model (sd-turbo), n = 30 per condition, one classifier; the in-house pipeline (float32, slicing) is not isolated; the lead has a post hoc origin.

## 9. Reproducibility

Scripts in `code/bin/` (each with a self-test): `h1-extract.py`, `h1-analyze.py`, `h1-controls-extract.py`, `h1-post.py`, `h1-vae-auc.py`, `h1-loco.py`, `h1-native.py`, `h1-scale.py`, `h1-resample.py`, `h1-owngen-make.py`, `h1-owngen-eval.py`, `h1-chain.py`, `h1r-explore.py`, `h2-sdfamily.py`, `h2-steps.py`, `h2-descriptors.py`, `archive-results.py`. Raw results in `results-archive/` (those of H1 and H1-R plus those of H2; SHA-256 in `MANIFEST.sha256`). Seed 20261111; sd-turbo generation with seeds from 20261112. sd-turbo images: tar with SHA-256 `8c02cf0ea1921c2dd47107ad6afb3a40306ddabed18884b7cb446294c992f9ab` (copy verified by the author outside the server). Git tags: `h1-criterios-s07…s21`, `h1-resultados-s22`, `h1-informe-final`, `h1-congelada`, `h1r-criterios`, `h1r-resultados`, `h2-criterios`, `h2-etapa0`, `h2-etapa1-listo`, `h2-etapa2-listo`, `h2-cerrada` (branch `h2-sdturbo`). H2 images: tar with SHA-256 `e49ffbb81504f233666563a0aad2bc7b857349b9df8885e3b370c4cde0ab53d8`; generation with `h1-owngen-make.py --seed-offset 200000 --new-prompts --steps {1,2,4}`. Environment: CPU, 3.8 GB of RAM; diffusers and torch versions in the generation manifest.

## Statements

- **Data and code availability.** Code, preregistrations and raw results (with SHA-256) are in the repository `GustavoHerraiz/ai-spectral-prereg`, branch `main`. The sd-turbo images are in a tar with the hash of §9. Images from third-party corpora are not redistributed; Synthbuster and Dresden are public. The provenance of COCO-512 and of the 8 `ai_*` sets is undocumented.
- **Verification of figures.** The figures of §3 and §4 were checked against the archived JSON files (`results-archive/`): H1-PRNU and initial L2 (`h1-analysis.json`), controls (`h1-post.json`), VAE (`h1-vae-auc.json`), LOCO (`h1-loco.json`), native (`h1-native.json`), scale (`h1-scale.json`), resampling (`h1-resample.json`), in-house AI and chain (`h1-owngen.json`, `h1-chain.json`) and H1-R. **Exception:** the probe v1 figures come from its preregistration and are not in `results-archive/`.
- **Use of an AI assistant.** The experimental design and decisions are the author's. An AI assistant (GitHub Copilot) helped write scripts, preregistrations and text, and in the exploratory analysis of §6.3; the author ran all experiments on their server and accepted the conclusions. The text, including this translation, requires external review before submission.
- **Conflicts of interest and funding.** No external funding. The author develops the Dragon image authentication system, whose v1 detector is mentioned in the probe.

## References

1. Manisha, Li, C.-T., Kotegar, K. A. *AI-Synthesized Image Detection: Source Camera Fingerprinting to Discern the Authenticity of Digital Images.* IEEE Access, 2025. DOI 10.1109/ACCESS.2025.3537558.
2. Manisha, Li, C.-T., Lin, X., Kotegar, K. A. *Beyond PRNU: Learning Robust Device-Specific Fingerprint for Source Camera Identification.* Sensors 22(20):7871, 2022. DOI 10.3390/s22207871.
3. Mallet, A., Méreur, A., Kuribayashi, M., Cogranne, R., Bas, P. *Simple Detection of AI-Generated Images based on Noise Correlation.* AMLDS 2025. DOI 10.1109/AMLDS63918.2025.11159445; HAL hal-05041276.
4. Bammey, Q. *Synthbuster: Towards Detection of Diffusion Model Generated Images.* IEEE Open Journal of Signal Processing 5:1–9, 2024. DOI 10.1109/OJSP.2023.3337714.
5. Gloe, T., Böhme, R. *The Dresden Image Database for Benchmarking Digital Image Forensics.* Journal of Digital Forensic Practice 3(2–4):150–159, 2010. DOI 10.1080/15567281.2010.531500.
6. Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014. arXiv:1405.0312.
7. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R. *Adversarial Diffusion Distillation (SD-Turbo).* arXiv:2311.17042, 2023.
8. Lukáš, J., Fridrich, J., Goljan, M. *Digital Camera Identification from Sensor Pattern Noise.* IEEE Trans. Inf. Forensics Secur. 1(2):205–214, 2006. DOI 10.1109/TIFS.2006.873602.
9. Corvi, R. et al. *On the Detection of Synthetic Images Generated by Diffusion Models.* ICASSP 2023. DOI 10.1109/ICASSP49357.2023.10095167.

*References 1–3 and 9 were checked against the PDFs or the repository metadata (1–3) and against correspondence with the authors (9, not included); 4, 5, 6, 7 and 8 were checked on the web against abstracts or source metadata (titles, authors, volume and DOI or arXiv). For 8, the volume, issue and first page were verified; the last page (214) and the venue of 6 (ECCV 2014) could not be read in the source and must be checked before submission.*

## Appendix A. Chronological record

`MANUSCRITO.md` §5.1–5.13: all stages, commands, commits and intermediate figures (in Spanish).
