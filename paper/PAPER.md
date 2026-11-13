# La separación espectral entre fotos reales e imágenes de IA se explica por la cadena de procesado y no por una huella de generación: un informe negativo preregistrado

**Autor:** Gustavo Herraiz. **Estado:** borrador de preprint, pendiente de revisión externa. **Rama:** `main` (base congelada: tag `h1-congelada`).
**English version:** `PAPER_EN.md` (traducción; esta versión en español prevalece).
**Registro cronológico completo y todas las cifras intermedias:** `MANUSCRITO.md` (Apéndice A). Preregistros: `PREREGISTRO_H1.md`, `PREREGISTRO_H1R.md`.

## Resumen

**Contexto.** Los detectores de imágenes sintéticas basados en espectro se evalúan a menudo con reales e imágenes de IA que difieren en formato, tamaño e historial de compresión, de modo que parte de su separación puede deberse al procesado y no al origen. **Objetivo.** Determinar cuánto de la separación entre reales e IA que logran dos rasgos espectrales propios (`res_spec`, `lf_eq`) sobrevive cuando se controla el procesado. **Métodos.** Estudio preregistrado (criterios y predicciones fijados antes de ver los datos; autotests con efecto inyectado y nulo; todo lo post hoc rotulado «exploratorio») con 205 fotos de móvil, 90 de cámaras Dresden, el benchmark Synthbuster [4] (700 IA, 7 familias) y 100 imágenes de un generador propio de historial conocido (sd-turbo, 512×512 nativo). Se valida entre corpus de reales (run3 ↔ Dresden) con IC95 por bootstrap. **Resultados.** (i) Con el mismo reescalado y formato en ambas clases la separación cae de ≈0,8–0,9 a ≈0,5–0,6 AUC y no generaliza entre corpus (`res_spec` 0,536; `lf_eq` 0,502). (ii) `res_spec` aguanta a resolución nativa (0,897) y tras imitar el procesado solo en la IA (0,856); con la IA propia llega a 0,942 si la cadena (paso bajo ×1,5 + JPEG q90 + q92) se aplica solo a la IA. (iii) Al aplicar **la misma cadena a los reales**, `res_spec` cae a 0,635 (umbral preregistrado 0,68; mín. IC inf. 0,544) y el modelo marca como «IA» a reales procesados (0,866). (iv) En una replicación exploratoria con 8 conjuntos de IA de procedencia no verificada, `res_spec` con cadena igualada da 0,42–0,56 en los 8. **Conclusión.** En este benchmark y con estos rasgos, la separación observada refleja la cadena de procesado y la escala, no una huella de generación. Alcance: rasgos propios y proxy, un benchmark y un generador propio confirmatorio; los reales se reducen ≥3× mientras la IA es nativa a 512 (asimetría de escala no resuelta). **No se refutan los detectores publicados.** Persisten dos cuestiones abiertas: un residual débil de `res_spec` (0,61–0,66) con sd-turbo y una discordancia sin explicar (§6.3). Una investigación preregistrada posterior (H2, §4.4) descartó el linaje SD del entrenamiento y la nitidez como causa; el desplazamiento se reproduce solo con 2 pasos de sd-turbo y su origen sigue sin identificarse.

**Palabras clave:** detección de imágenes sintéticas, huella espectral, preregistro, resultado negativo, cadena de procesado, PRNU.

## 1. Introducción

Los detectores de imágenes generadas se suelen validar con reales y sintéticas de procedencia distinta (formato, tamaño, compresión). Si ambas clases difieren en su historial, un clasificador puede aprender el historial. Este trabajo parte de la réplica de las ideas de Manisha et al. [1, 2] (huella L1) y Mallet et al. [3] (correlaciones de ruido entre canales) y se pregunta qué parte de la separación sobrevive al control del procesado. Se plantean dos hipótesis preregistradas:

- **H1-PRNU:** la presencia del patrón de ruido de sensor del registro es señal de «real».
- **H1-L2:** una huella de baja/alta frecuencia separa reales de IA de familias no vistas tras igualar el procesado.

Se publica el resultado negativo con el mismo detalle que habría tenido uno positivo.

## 2. Métodos

### 2.1 Datos

| Corpus | n | Descripción |
|---|---|---|
| `run3` | 205 reales, 41 dispositivos (≤5 por dispositivo) | fotos nativas de móvil, JPEG, lado menor ≥2000 px |
| Dresden `nat` | 90 reales (≤10 por cámara, semilla 20261111) | JPEG de cámara |
| Synthbuster | 700 IA, 7 familias | 100 por familia; PNG salvo firefly (JPEG, 1792 px) |
| COCO-512 | 50 reales | reescaladas a 512×512, preparación no documentada |
| IA propia (sd-turbo) | 100 | 512×512 PNG nativo, 2 pasos, guidance 0, float32 con attention/VAE slicing (H1-D3); manifiesto con SHA-256 y versiones |
| 8 conjuntos `ai_*` | 50 cada uno | procedencia **no verificada** (exploratorio, §5.5) |

El detector v1, el registro PRNU de 81 K, v3 y el código existente son de solo lectura.

### 2.2 Rasgos y clasificador

`res_spec`: log-potencia radial del residuo (mediana 3×3, gris), 64 bins logarítmicos en [0,004; 0,5] ciclos/px. `lf_eq`: vector L1-proxy de 32 bins tras ecualizar el procesado (256 + JPEG Q90). Clasificador: estandarización + regresión logística L2 balanceada. Métrica: AUC (macro por familia en LOFO; en validación cruzada entre corpus, media de los dos sentidos run3→Dresden y Dresden→run3). IC95 por bootstrap sobre dispositivos (reales) e imágenes (IA), semilla 20261111.

### 2.3 Cadenas de procesado

A0: sin procesar. A1: JPEG q92. A5: paso bajo (Lanczos ×1,5 arriba y abajo) + JPEG q90 + q92. Reales: R0 (reducción a lado menor 512, recorte alineado a 8), R1 (R0 + q92), R5 (R0 con la cadena A5).

### 2.4 Diseño preregistrado

Hipótesis, umbrales y predicciones (con probabilidad) se fijaron y se etiquetaron en git antes de cada ejecución (`h1-criterios-s07…s21`, `h1r-criterios`). Autotests con efecto inyectado y nulo antes de cada ejecución real. Desviaciones registradas antes de ejecutar (H1-D1 a D3). Umbrales de veredicto en validación cruzada entre corpus: «aguanta» si media ≥0,75 y mín. IC inf. ≥0,60; «artefacto» si media <0,68; si no, «indeterminada». Se aplica a todas las comparaciones (c) de §4.

### 2.5 Análisis de cadena (§21)

(c) **primario, cadena igualada en ambos lados:** modelo entrenado con Synthbuster A5 y R5 de un corpus; prueba con IA propia A5 frente a R5 del otro. (a) diagnóstico: modelo A5/R1; AUC IA propia A5 > IA propia A1 (mismas imágenes). (b) diagnóstico: modelo A5/R1; AUC R5 > R1 (mismos reales). (a) y (b) altos indican que el modelo detecta la cadena.

## 3. Resultados de las etapas previas

| Etapa | Resultado | Lectura |
|---|---|---|
| Sonda v1 (run3) | `jpegPhaseRange` 0,612 [0,570; 0,655]; `phaseCoherence` 0,611 | MIXTO; variación entre dispositivos reales > diferencia real–IA |
| H1-PRNU (995 imágenes) | reales con K 0,931 [0,905; 0,954]; cámara ausente 0,587 [0,494; 0,675]; IA: firefly 0,56 sobre umbral, resto 0 | el 56 % de firefly es un artefacto JPEG genérico (53/100 ganadas por una sola K) |
| H1-L2 inicial (LOFO) | `lf_eq` 0,799 [0,776; 0,821]; `res_spec` 0,901 [0,885; 0,915] | línea base de bytes 0,993: confusión de formato |
| Controles de formato | `res_spec` 0,774–0,916; `lf_eq` 0,581–0,877; JPEG real vs firefly (mismo formato) `res_spec` 0,924 | veredicto parcial; predicción «`res_spec` se hunde» refutada |
| Control VAE (exploratorio, 50 reales) | VAE vs original 0,586 (`res_spec`); B ≈ C | el decodificador (este VAE, a 512) no explica la separación |
| LOCO, pipeline común a 512 | `res_spec` 0,536; `lf_eq` 0,502 | artefacto de corpus (no transfiere) |
| LOCO nativo (run3↔Dresden) | `res_spec` 0,897; `lf_eq` 0,733 | `res_spec` aguanta; predicción refutada |
| Dosis-respuesta de escala | `res_spec` 0,897→0,781 con reducción 1→512; `lf_eq` 0,733→0,509 | `lf_eq` depende de escala; `res_spec` no se explica por escala |
| Remuestreo + doble JPEG solo en IA | `res_spec` 0,856 (mín. IC inf. 0,797) | «persiste» (**conclusión superada por §4, ver erratum §7**) |

Detalle en `MANUSCRITO.md` §5.1–5.9.

## 4. Resultados principales

### 4.1 IA propia, cadena solo en la IA (§20, n=100)

| Brazo | `res_spec` (media cruzada) | Veredicto |
|---|---|---|
| A5/R1 (primario) | 0,942 [mín. IC inf. 0,907] | se confirma (predicción p≈0,4 refutada) |
| A1/R1 | 0,509 | azar (Synthbuster: 0,781) |
| A0/R0 | 0,704 | indeterminada |

`lf_eq`: 0,885 en A5/R1 pero 0,251 (A0/R0) y 0,219 (A1/R1), invertido: no es una señal de generación.

### 4.2 Control de cadena (§22, sd-turbo)

| Rasgo | Análisis | run3→Dresden | Dresden→run3 | Media | Veredicto |
|---|---|---|---|---|---|
| `res_spec` | **(c) cadena igualada (primario)** | 0,661 [0,586; 0,739] | 0,608 [0,544; 0,669] | **0,635** | **artefacto de cadena** |
| `res_spec` | (a) IA A5 vs A1 | 0,640 [0,616; 0,673] | 0,643 [0,618; 0,678] | 0,642 | cadena no detectada |
| `res_spec` | (b) reales R5 vs R1 | 0,904 [0,842; 0,962] | 0,827 [0,790; 0,866] | 0,866 | cadena detectada |
| `lf_eq` | (c) | 0,242 [0,163; 0,325] | 0,314 [0,257; 0,371] | 0,278 | artefacto de cadena |

Predicciones: (c) artefacto p≈0,6, cumplida; (b) detectada p≈0,6, cumplida; (a) detectada p≈0,8, **refutada**.

### 4.3 Replicación exploratoria (H1-R, procedencia no verificada)

`res_spec` con cadena igualada (c), media cruzada: Flux schnell 0,472; GPT-image 0,461; Kandinsky 2.2 0,525; Nano Banana 0,475; Nano Banana (generada, JPEG previo y reescalado conocidos) 0,514; PixArt-Σ 0,421; SDXL 0,559; Würstchen 0,429. Las 8 son «artefacto de cadena». `lf_eq` (c): 0,433–0,497. A5 vs A1 (a): 0,845–0,964 en las 8. Predicciones: mayoría <0,68, cumplida (8/8); alguna «persiste» (p≈0,3), no ocurrió; Nano Banana distinta del resto (p≈0,5), refutada.

### 4.4 H2: investigación de la pista de sd-turbo (rama `h2-sdturbo`, `PREREGISTRO_H2.md`)

Sin cadena (A1), sd-turbo ya puntúa «IA» con el modelo entrenado en Synthbuster A5 vs R1 (AUC 0,899), cosa que no hacen las 8 carpetas de §4.3. Tres análisis preregistrados, con criterios y predicciones fijados antes de ejecutar:

| etapa | pregunta | resultado | veredicto |
|---|---|---|---|
| 0 | ¿linaje Stable Diffusion del entrenamiento? | sin SD1+SD2+SDXL: 0,919 (mín. IC inf. 0,867); controles 0,884–0,904 | no explicada por linaje SD |
| 1 | ¿se reproduce (30 imágenes nuevas) y depende de los pasos? | 2 pasos 0,816 (0,704); 1 paso 0,375; 4 pasos 0,572; S(1)−S(4) = −0,197 [−0,311; −0,086] | se reproduce; depende de los pasos, pero **no monótono** |
| 2 | ¿lo sigue la nitidez o la alta frecuencia? (exploratoria) | Spearman 0,094 (`d_lap`) y 0,163 (`d_hf`), ICs incluyen 0 | ningún descriptor |

El desplazamiento es real y reproducible, específico de la configuración (2 pasos) y no monótono, lo que no apoya un rasgo general de destilación; **su causa no se identificó**. Predicciones: etapa 0 (linaje, p≈0,35) no cumplida; etapa 1 (réplica p≈0,75; pasos p≈0,35) cumplidas; etapa 2 (ninguno, p≈0,30) cumplida. Un solo modelo, n = 30 por condición. Un `.tar` de las 90 imágenes tiene SHA-256 `e49ffbb81504f233666563a0aad2bc7b857349b9df8885e3b370c4cde0ab53d8`.

## 5. Estatuto de la evidencia

- **Confirmatoria (criterios fijados antes):** LOCO, nativo, escala, IA propia (§4.1), control de cadena (§4.2).
- **Preregistrada (criterios fijados antes), con alcance de un solo modelo:** H2 etapas 0 y 1 (§4.4).
- **Exploratoria:** control VAE, replicación H1-R (§4.3), diagnóstico de §6.3 (§5.13 del manuscrito), H2 etapa 2 (§4.4).
- Los 8 conjuntos de §4.3 no cuentan como replicación confirmatoria: sin procedencia, con posible reescalado previo y sin control de prompts.

## 6. Discusión

### 6.1 Qué mide `res_spec`
La aparente huella de `res_spec` se reproduce solo cuando la cadena se impone a una clase. Con la cadena en ambos lados cae de 0,942 a 0,635, y los reales procesados puntúan «IA» (0,866). Es coherente con que el rasgo detecta el procesado (paso bajo + doble JPEG), no la generación.

### 6.2 Escala y procesado
`lf_eq` es función de la escala (0,733→0,509 con la reducción). `res_spec` decae menos, pero la asimetría real/IA (reales reducidos ≥3×, IA nativa a 512) nunca se rompió: no hay reales nativos de ~512. Mientras no exista, cualquier AUC de este diseño mezcla escala con origen.

### 6.3 Cuestiones abiertas
1. **Residual débil.** (c) 0,61–0,66 con IC inferior >0,5 en sd-turbo no se descarta; está bajo el umbral preregistrado y no se reproduce en las 8 carpetas (0,42–0,56).
2. **Discordancia.** En sd-turbo, A5 vs A1 es 0,64 pero en las 8 carpetas es 0,85–0,96. Análisis post hoc: sd-turbo ya puntúa «IA» sin la cadena (A1: +1,5 a +2,6, frente a −2,7 a −0,6 en las demás) y la cadena solo añade ≈+1 (≈+3 en las demás). La hipótesis previa de «dispersión por contenido» se **refuta**: el IQR de sd-turbo (2,1–2,3) es similar al de las demás (1,3–2,2); la diferencia es de posición. **H2 etapa 0 (preregistrada, `PREREGISTRO_H2.md` §7):** reentrenar sin las familias SD1, SD2 y SDXL no reduce el desplazamiento (A1 vs R1: 0,899 → 0,919; controles 0,884–0,904), de modo que el linaje Stable Diffusion del entrenamiento **no** lo explica. **Etapa 1 (`PREREGISTRO_H2.md` §9):** con 30 imágenes nuevas por condición, el desplazamiento se reproduce con 2 pasos (0,816; mín. IC inf. 0,704) pero **no es monótono en los pasos** (1 paso 0,375; 4 pasos 0,572), por lo que no apoya un rasgo de destilación y es específico de la configuración; un solo modelo, n = 30. **Etapa 2 (`PREREGISTRO_H2.md` §11):** ni la nitidez (`d_lap`, rho 0,09) ni la fracción de alta frecuencia (`d_hf`, rho 0,16) siguen al puntaje; la hipótesis de textura queda refutada y **la causa sigue sin identificarse**. Causas no distinguibles: propiedad de generación (2 pasos, destilado), pipeline propio (float32, slicing) o historiales de las otras carpetas que borran el rasgo. Es una pista que requiere preregistro, no un hallazgo.

### 6.4 Relación con la literatura
Los resultados no contradicen a [1, 3]: evalúan rasgos propios o proxy en un benchmark y entrenamiento distintos. Sí señalan que, en este diseño, controlar el procesado elimina la separación.

## 7. Conclusiones

1. Con `res_spec` y `lf_eq`, la separación real/IA en este benchmark **se explica por la cadena de procesado y la escala**, no por una huella de generación.
2. La prueba decisiva: con la cadena igualada en ambos lados, `res_spec` cae a 0,635 (<0,68), y los reales procesados puntúan «IA» (0,866).
3. El patrón no se contradice con 8 conjuntos adicionales de procedencia no verificada (0,42–0,56), de forma exploratoria.
4. H1-PRNU y H1-L2 quedan **no apoyadas**.
5. **Erratum.** La conclusión de §5.9 («persiste» con 0,856) aplicó la cadena solo a la IA y no igualaba el procesado; queda superada por §4.2.
6. **H2.** El puntaje «IA» de sd-turbo sin cadena no se explica por el linaje SD del entrenamiento ni por nitidez o alta frecuencia; se reproduce solo con 2 pasos y no es monótono en los pasos. Es una propiedad de esa configuración, no una huella general de «IA», y su causa queda **sin identificar** (un solo modelo).
7. Para detectores espectrales, la lección metodológica es: igualar la cadena de procesado en ambas clases y verificar que el modelo no marca como IA a reales procesados.

## 8. Limitaciones

- Un benchmark (Synthbuster, firefly en JPEG y a 1792 px) y un generador propio confirmatorio (sd-turbo, n=100, float32 con slicing); las 8 carpetas adicionales tienen procedencia no verificada.
- Asimetría de escala sin romper; reales con ≤5 (run3) o ≤10 (Dresden) imágenes por dispositivo; solo 3 corpus de reales; COCO-512 sin pipeline documentado.
- Bootstrap por imagen no modela dependencia entre imágenes del mismo dispositivo o escena en COCO y VAE.
- Un VAE de SD a 512; no representa a DALL·E, Midjourney ni Firefly.
- Los resultados se refieren a estos rasgos y este entrenamiento; no refutan detectores publicados.
- Preregistros etiquetados en git pero no firmados criptográficamente.
- Un residual débil (0,61–0,66) no se excluye.
- H2: un solo modelo (sd-turbo), n = 30 por condición, un clasificador; el pipeline propio (float32, slicing) no se aísla; la pista tiene origen post hoc.

## 9. Reproducibilidad

Scripts en `code/bin/` (autotest cada uno): `h1-extract.py`, `h1-analyze.py`, `h1-controls-extract.py`, `h1-post.py`, `h1-vae-auc.py`, `h1-loco.py`, `h1-native.py`, `h1-scale.py`, `h1-resample.py`, `h1-owngen-make.py`, `h1-owngen-eval.py`, `h1-chain.py`, `h1r-explore.py`, `h2-sdfamily.py`, `h2-steps.py`, `h2-descriptors.py`, `archive-results.py`. Resultados crudos en `results-archive/` (los de H1 y H1-R más los de H2; SHA-256 en `MANIFEST.sha256`). Semilla 20261111; generación sd-turbo con semillas desde 20261112. Imágenes sd-turbo: tar con SHA-256 `8c02cf0ea1921c2dd47107ad6afb3a40306ddabed18884b7cb446294c992f9ab` (copia verificada por el autor fuera del servidor). Etiquetas git: `h1-criterios-s07…s21`, `h1-resultados-s22`, `h1-informe-final`, `h1-congelada`, `h1r-criterios`, `h1r-resultados`, `h2-criterios`, `h2-etapa0`, `h2-etapa1-listo`, `h2-etapa2-listo`, `h2-cerrada` (rama `h2-sdturbo`). Imágenes de H2: tar con SHA-256 `e49ffbb81504f233666563a0aad2bc7b857349b9df8885e3b370c4cde0ab53d8`; generación con `h1-owngen-make.py --seed-offset 200000 --new-prompts --steps {1,2,4}`. Entorno: CPU, 3,8 GB de RAM; versiones de diffusers y torch en el manifiesto de generación.

## Declaraciones

- **Disponibilidad de datos y código.** Código, preregistros y resultados crudos (con SHA-256) en el repositorio `GustavoHerraiz/ai-spectral-prereg`, rama `main`. Las imágenes de sd-turbo están en un tar con el hash de §9. Las imágenes de los corpus de terceros no se redistribuyen; Synthbuster y Dresden son públicos. La procedencia de COCO-512 y de los 8 conjuntos `ai_*` no está documentada.
- **Verificación de cifras.** Las cifras de §3 y §4 se contrastaron con los JSON archivados (`results-archive/`): H1-PRNU y L2 inicial (`h1-analysis.json`), controles (`h1-post.json`), VAE (`h1-vae-auc.json`), LOCO (`h1-loco.json`), nativo (`h1-native.json`), escala (`h1-scale.json`), remuestreo (`h1-resample.json`), IA propia y cadena (`h1-owngen.json`, `h1-chain.json`) y H1-R. **Excepción:** las cifras de la sonda v1 proceden de su preregistro y no están en `results-archive/`.
- **Uso de asistente de IA.** El diseño experimental y las decisiones son del autor. Un asistente de IA (GitHub Copilot) colaboró en la escritura de scripts, preregistros y texto, y en el análisis exploratorio de §6.3; el autor ejecutó todos los experimentos en su servidor y aceptó las conclusiones. El texto requiere revisión externa antes de enviarse.
- **Conflictos de intereses y financiación.** Ninguna financiación externa. El autor desarrolla el sistema de autenticación de imágenes Dragon, cuyo detector v1 se menciona en la sonda.

## Referencias

1. Manisha, Li, C.-T., Kotegar, K. A. *AI-Synthesized Image Detection: Source Camera Fingerprinting to Discern the Authenticity of Digital Images.* IEEE Access, 2025. DOI 10.1109/ACCESS.2025.3537558.
2. Manisha, Li, C.-T., Lin, X., Kotegar, K. A. *Beyond PRNU: Learning Robust Device-Specific Fingerprint for Source Camera Identification.* Sensors 22(20):7871, 2022. DOI 10.3390/s22207871.
3. Mallet, A., Méreur, A., Kuribayashi, M., Cogranne, R., Bas, P. *Simple Detection of AI-Generated Images based on Noise Correlation.* AMLDS 2025. DOI 10.1109/AMLDS63918.2025.11159445; HAL hal-05041276.
4. Bammey, Q. *Synthbuster: Towards Detection of Diffusion Model Generated Images.* IEEE Open Journal of Signal Processing 5:1–9, 2024. DOI 10.1109/OJSP.2023.3337714.
5. Gloe, T., Böhme, R. *The Dresden Image Database for Benchmarking Digital Image Forensics.* Journal of Digital Forensic Practice 3(2–4):150–159, 2010. DOI 10.1080/15567281.2010.531500.
6. Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014. arXiv:1405.0312.
7. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R. *Adversarial Diffusion Distillation (SD-Turbo).* arXiv:2311.17042, 2023.
8. Lukáš, J., Fridrich, J., Goljan, M. *Digital Camera Identification from Sensor Pattern Noise.* IEEE Trans. Inf. Forensics Secur. 1(2):205–214, 2006. DOI 10.1109/TIFS.2006.873602.
9. Corvi, R. et al. *On the Detection of Synthetic Images Generated by Diffusion Models.* ICASSP 2023. DOI 10.1109/ICASSP49357.2023.10095167.

*Las referencias 1–3 y 9 se contrastaron con los PDF o los metadatos del repositorio (1–3) y con la correspondencia con los autores (9, no incluida); la 4, la 5, la 6, la 7 y la 8 se contrastaron en la web con resúmenes o metadatos de la fuente (títulos, autores, volumen y DOI o arXiv). De la 8 se comprobó el volumen, el número y la primera página; la página final (214) y el lugar de publicación de la 6 (ECCV 2014) no se pudieron leer en la fuente y deben revisarse antes del envío.*

## Apéndice A. Registro cronológico

`MANUSCRITO.md` §5.1–5.13: todas las etapas, comandos, commits y cifras intermedias.
