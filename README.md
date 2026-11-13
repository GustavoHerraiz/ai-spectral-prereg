# ai-spectral-prereg

**ES** — Informe negativo preregistrado: ¿separa una huella física/espectral (PRNU, frecuencias altas) las fotos reales de las imágenes generadas por IA? Resultado: no se sostiene; el efecto observado era un artefacto de la cadena de procesado y la escala. Pendiente de revisión externa.

**EN** — Preregistered negative report: does a physical/spectral fingerprint (PRNU, high frequencies) separate real photos from AI-generated images? Result: not supported; the observed effect was an artifact of the processing chain and scale. Pending external review.

## Leer / Read
- [Paper (español, versión de referencia)](paper/PAPER.md)
- [Paper (English, not native-reviewed)](paper/PAPER_EN.md)
- [Registro cronológico / chronological record](paper/MANUSCRITO.md)

## Contenido / Contents
- `prereg/` — preregistros H1, H1-R, H2 y anteriores, con resultados / preregistrations with results.
- `code/` — scripts (cada uno con autotest) y configuración / scripts (each with self-test) and config. Las rutas por defecto `/home/gustavo/...` son del servidor del autor; pásense `--root`/`--out` propios.
- `results-archive/` — resultados en bruto + `MANIFEST.sha256` / raw results plus SHA-256 manifest.

## Cronología de los preregistros / Preregistration chronology
La fecha de cada preregistro la fijan tags de git (`h1-criterios-*`, `h1r-criterios`, `h2-criterios`, `h2-etapa0`, `h2-etapa1-listo`, `h2-etapa2-listo`, `h2-cerrada`…) del repositorio de investigación original, que no viajan a este repo. Los hashes de los resultados están en `results-archive/MANIFEST.sha256`. Los datos de imágenes no se incluyen (imágenes de terceros).
Preregistration dates are fixed by git tags in the original research repository, which do not carry over here. Image data is not included (third-party images).

### Commits originales / Original commits
Repo original / original repo: `GustavoHerraiz/dragon3-detector-research` (rama/branch `h1-replicacion`; privado/private).

| tag | commit | fecha / date |
|---|---|---|
| `h1-criterios-s07` | `0799ad1` | 2026-11-11 |
| `h1-criterios-s09` | `7a852e7` | 2026-11-12 |
| `h1-criterios-s11` | `ab6e3f3` | 2026-11-12 |
| `h1-criterios-s13` | `e78e3ca` | 2026-11-12 |
| `h1-criterios-s15` | `1642200` | 2026-11-12 |
| `h1-criterios-s17` | `48ec33b` | 2026-11-12 |
| `h1-criterios-s19` | `31ce462` | 2026-11-12 |
| `h1-criterios-s21` | `e11948a` | 2026-11-12 |
| `h1-congelada` | `b8038bd` | 2026-11-12 |
| `h1-resultados-s22` | `9320bfc` | 2026-11-12 |
| `h1-informe-final` | `a9c5aa0` | 2026-11-12 |
| `h1r-criterios` | `ad7e9a5` | 2026-11-12 |
| `h1r-resultados` | `3922282` | 2026-11-12 |
| `h2-criterios` | `9f8fbd5` | 2026-11-12 |
| `h2-etapa0` | `0e72e79` | 2026-11-12 |
| `h2-etapa1-listo` | `fc5c6f5` | 2026-11-12 |
| `h2-etapa2-listo` | `ba265f1` | 2026-11-13 |
| `h2-cerrada` | `81f8f1f` | 2026-11-13 |
| `h2-paper-v4` | `d7d7d51` | 2026-11-13 |
| `h2-paper-en` | `5306529` | 2026-11-13 |

## Licencia / License
Texto / text: CC BY 4.0. Código / code: MIT. Ver / see `LICENSE`.

## Cita / Cite
Gustavo Herraiz. Preprint, pendiente de revisión externa / pending external review.
