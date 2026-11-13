# Preregistro: sonda de procesado de Detector v1 (diseño factorial pareado)

Estado: **borrador, no congelado**. Línea auxiliar de `l1-manisha2025`. No modifica Detector v1, su sello ni sus resultados (`CIERRE_V1.md`).
Código: `bin/v1-pipeline-probe.mjs` (extracción), `bin/v1-pipeline-probe-analyze.py` (análisis).

## 1. Pregunta e hipótesis

Las dos features congeladas de v1 (`jpegPhaseRange`, `phaseCoherence`) separan IA de COCO (AUC 0,68–0,84 por generador, `results/tpr-original-ai`). La pregunta es qué parte de esa separación se debe al **historial de procesado** (JPEG previo, reescalado) y no al origen generativo.

- **H-PROC.** Aplicar a fotos reales un historial "JPEG de origen + reescalado" desplaza ambas features de forma comparable a la separación IA/COCO de v1. Si se aplica el mismo procesado a las dos clases, la separación IA/real cae.
- **H-RES (alternativa).** La separación IA/real se mantiene con procesado idéntico en ambas clases. Aun así, no se atribuiría al generador (ver sección 10).
- **Exploratoria, H-GEN.** Si hay señal residual, difiere entre familias generativas.

Las hipótesis se formularon **después** de ver resultados de v1 (sección 2). Se declara como generación post hoc. Lo que se registra antes de ver datos de esta sonda son el diseño, las reglas de decisión y los umbrales.

## 2. Evidencia previa (post hoc, ya vista)

Recalculada sobre JSON persistidos, sin imágenes:

- Los 5 generadores se desplazan en la misma dirección. Con las 2 features, la identificación del generador (5 clases) da 24 % de acierto con 20 % por azar.
- Las features son casi independientes (Spearman 0,07–0,18; AUC en CV 0,77 y 0,77 por separado, 0,83 juntas).
- `np.roll(32,32)` mantiene la AUC dentro de ±0,06 (`results/detector-v1-confounder-audit/translation.json`). La composición no explica la fase.
- Nano-Banana (edición y generación) tiene AUC ≈ 0,60 en ambos casos. Los nombres `_512` sugieren reescalado, **no verificado**. GPT-4o (0,83) no encaja limpio con esa idea.
- `phaseCoherence` usa la banda 0,02–0,35 ciclos/píxel y `jpegPhaseRange` mide la sensibilidad a la alineación de la rejilla JPEG. Ambas son plausiblemente sensibles al reescalado y a la compresión previa. No está demostrado.

### 2.1 Hipótesis ya sometidas a control en el repositorio (leídas antes de congelar)

Todos los informes son exploratorios y reconocen que **cámara/generador e historia JPEG siguen sin control**. Esa es la laguna que cubre esta sonda.

| Hipótesis | Prueba previa | Resultado | Lo que NO controló |
|---|---|---|---|
| Solo es nitidez | `results/degree-variance-sharpness-matched-20261013T234345Z.md` (caliper ≤ 1 Sobel, 181 pares, cobertura 74,5 %) | win-rate pareado 0,762 [0,696, 0,823]; AUC subset 0,743 | corpus, generador, cámara, historia JPEG |
| Es desenfoque | `degree-variance-blur-crossed-20261013T223211Z.md` | AUC 0,85 sin blur; cae a 0,56 en σ=3 y se invierte (0,38) en σ=8; el igualado de nitidez media falló la tolerancia | causa del cambio de régimen |
| Es asociación al azar | `results/label-permutation/2026-10-14T19-16-13-288Z/INFORME_PERMUTACION.md` | gap 0,220 frente a nulo SD 0,018; p MC 0,001 (n=636) | generador, JPEG, contenido, fuente |
| Es ruido de cámara / textura fina | `results/jpeg-spectrum-frozen-denoising/*/INFORME_CONTROL_CONGELADO.md` (NLM sobre reales) | `jpegPhaseRange` cae 0,10–0,21 de AUC y se mantiene > 0,5 en 4 de 5 generadores (SDXL 0,45); el espectro de alta frecuencia se invierte | NLM también cambia textura y estructura |
| Es un solo generador | `results/multi-generator-signal-panel/2026-10-19T22-05-31-664Z/INFORME_MULTI_GENERADOR.md` | `degreeVariance`: SD1.5 0,56 frente a SDXL 0,77: depende del generador | procesado de las reales |
| Es composición | `translation.json` (sección 2) | ±0,06 | — |

Consecuencia para esta sonda: nitidez, desenfoque y azar ya no son candidatas principales. Un veredicto EXPLICADO_POR_PROCESADO no contradiría esos controles, porque ninguno igualó el historial de procesado de la fuente. Esta tabla se escribió antes de ver ningún resultado de la sonda.

## 3. Diseño

Factorial 2×2 **dentro de imagen** (cada imagen genera sus variantes):

| Variante | JPEG previo (Q) | Reescalado ×0,5 | Descripción |
|---|---|---|---|
| C0 | no | no | recorte central 512 nativo |
| C1 | sí | no | JPEG y recorte 512 |
| R0 | no | sí | ventana 768 → Lanczos3 (×2/3) → 512 |
| R1 | sí | sí | JPEG → ventana 768 → Lanczos3 (×2/3) → 512 (emula JPEG de origen + reescalado) |

- Misma ventana, alineada a múltiplos de 16, para todas las variantes. El factor de reescalado es idéntico en ambas clases.
- JPEG: libvips vía `sharp`, calidad **75**, submuestreo 4:2:0. Es un valor fijo. Los parámetros reales de COCO son desconocidos; la sonda prueba el **mecanismo**, no reproduce COCO.
- R0/R1 solo existen si el lado menor ≥ 768. Nunca se amplía. Las imágenes con 512 ≤ lado < 768 solo generan C0/C1. Las de lado < 512 quedan fuera.
- Las variantes se guardan solo en memoria (con SHA-256 de cada una en el JSONL).

## 4. Datos y muestreo

- Reales: RAISE-1k (PNG). IA: Synthbuster, 8 familias (SD 1.3 + 1.4 = `sd1`). Rutas por defecto en el script.
- **Inventario previo** (`--inventory`): solo metadatos (tamaño, formato, profundidad). Fija qué familias son elegibles para R0/R1. La elegibilidad se decide por la regla de tamaño, no por resultados.
- Muestreo: semilla 20261111; 200 reales y 100 imágenes por familia, aleatorio con semilla. Si la familia tiene ≥ 30 imágenes con lado ≥ 768, se muestrea solo entre ellas (diseño pareado); si no, entre las de lado ≥ 512 y la familia solo aporta C0/C1. Se guarda `sample.json`, con su SHA-256 en `meta.json`.
- **Conjunto primario (pareado):** imágenes con las 4 variantes. Se incluyen familias con ≥ 30 imágenes pareadas. Las demás solo aparecen en los análisis de C0/C1 (secundarios).

## 5. Extractor

`extractFeatures` de `bin/evaluate-mobile-v1.mjs`, **sin modificarlo**. Antes de procesar, el script verifica el manifiesto, el sello y los hashes de las fuentes congeladas (`loadFrozen`) y registra los hashes. Features: `[jpegPhaseRange, phaseCoherence]`. No se usan los modelos ni los umbrales P96: el análisis es **univariante**.

## 6. Resultados

**Primarios** (por feature, dirección fija "IA-alta", la de v1):
- **A_match(arm):** AUC macro por familia, IA contra reales con la misma variante, para las 4 variantes.
- **A_mis:** AUC macro de real R1 contra IA C0 (emula la situación COCO vs IA nativa).
- **A_pipe:** AUC de reales R1 contra las mismas reales C0 (solo efecto del procesado; se reporta orientada, `max(A, 1−A)`, y con signo).

**Secundarios:** efecto pareado de cada factor, por clase (mediana de Δ, IC95 bootstrap, d_z, Wilcoxon), diferencia de efectos entre clases (Mann-Whitney), d de Cohen IA/real en C0, AUC por familia, AUC de bytes del archivo fuente por familia (diagnóstico de sesgo de formato), regresión logística 2D en CV (exploratoria).

## 7. Análisis estadístico

- IC95%: bootstrap percentil, B = 2000, semilla 20261111, remuestreando imágenes (reales y cada familia) y usando los mismos índices en todas las variantes.
- AUC macro: media simple de AUC por familia. Una limitación declarada es que las mismas reales se reutilizan en todas las familias.
- No se ajusta por multiplicidad: los resultados primarios están acotados por las reglas de la sección 8.

## 8. Reglas de decisión (fijadas antes de ejecutar)

Se aplican **por separado a cada feature**:

| Veredicto | Condición |
|---|---|
| EXPLICADO_POR_PROCESADO | media de A_match en las 4 variantes ≤ 0,60 **y** A_mis ≥ 0,70 **y** A_pipe orientada (R1) ≥ 0,65 |
| SENAL_RESIDUAL | A_match ≥ 0,65 en **cada** variante **y** límite inferior del IC95 > 0,55 en cada una |
| MIXTO | cualquier otro caso |

Los umbrales provienen de la escala de v1 (0,68–0,84) y no se ajustan después. Que dos features den veredictos distintos es un resultado válido y se reporta tal cual.

## 9. Controles y autotests

- `v1-pipeline-probe.mjs --self-test` (7 pruebas): caja alineada, muestreo determinista, agrupación SD, equivalencia de C0 con el recorte central de v1, equivalencia de R0 con Lanczos3 de la imagen entera, determinismo del JPEG, particiones por shard y reanudación.
- `v1-pipeline-probe-analyze.py --self-test` (6 pruebas): AUC contra fuerza bruta, tres escenarios sintéticos con efecto conocido (procesado, residual, nulo) que deben dar el veredicto esperado, exclusión de familias pequeñas y rechazo de claves duplicadas.
- Control interno: A_pipe sobre pares de la misma foto. El contenido está controlado por diseño.

## 10. Limitaciones declaradas

- Reales (RAISE, Nikon D7000) y fakes (Synthbuster) tienen contenido y fuente distintos. SENAL_RESIDUAL **no** prueba un efecto del generador: indica solo que no se explica por los dos factores de procesado probados.
- C0 cubre 512 px nativos y R0/R1 cubren 768 px nativos (campo de visión distinto). La escala del contenido difiere entre reales (12 MP) y IA (1 MP).
- Se prueban un único nivel de JPEG (Q75), un único factor de reescalado y un solo kernel. No cubre el historial real de COCO.
- Según el inventario: SD 1.x (512 px) y GLIDE (256 px) no entran en el análisis primario. GLIDE ni siquiera cumple ≥ 512. SD 2 y SDXL solo parcialmente.
- **Las reales RAISE-1k ya están reducidas** (1258×833, ≈ 3,9× respecto a los 4928×3264 del sensor, kernel desconocido). R0/R1 es un segundo remuestreo, no el primero. El efecto del remuestreo previo de las reales no puede aislarse con este corpus.
- Univariante: el resultado no dice nada sobre v1 como detector completo (modelos P96).
- Muestra pequeña (200 / 100): los IC son amplios.
- Conversión 16→8 bit de libvips en caso de PNG de 16 bits: se registra la profundidad por imagen.

## 11. Registro de desviaciones

- Carrera de arranque (17:05Z): dos shards lanzados a la vez; el shard 0 leyó `sample.json` a medio escribir y abortó. Sin efecto en datos; el shard 0 se relanzó solo. Corrección de código pendiente.
- Cambio de diseño **previo al congelado**, tras el inventario: la ventana de reescalado pasa de 1024 (×0,5) a 768 (×2/3), porque los reales miden 1258×833 (lado menor 833 < 1024) y no podían generar R0/R1. Se añade además el muestreo preferente entre imágenes reescalables.

## 11b. Corrida run2-movil y run3-multidispositivo (exploratorias, registradas a posteriori)

Ambas se ejecutaron sin congelar el preregistro; todo resultado es exploratorio.

**run3-multidispositivo.** Criterios fijados antes de ver resultados (commit `4f48df1`, `bin/build-native-reals.py`): solo carpetas `nat`; JPEG; lado menor ≥ 2000 px, sin reescalar; ≤ 5 fotos por dispositivo (semilla fija); ≥ 8 dispositivos. Resultado del constructor: 205 fotos, 41 dispositivos. IA: 100 por familia (6 familias). Reglas de decisión de la sección 8, sin cambios.

**Desviaciones:**
- Los dos shards se lanzaron con 20 s de diferencia y el shard 1 abortó por la carrera de `sample.json` (hash distinto). El shard 0 completó 453 imágenes; el shard 1 se relanzó después (452). Los datos no se mezclaron entre muestras distintas; el análisis con la mitad de los datos (103 reales) se descartó y solo cuenta el análisis con las 905 imágenes.
- Los resultados no están versionados en el repositorio (solo en el servidor).

**Resultado (905 registros, 205 reales):** ambas características MIXTO. `jpegPhaseRange` AUC emparejada C0 0,612 [0,570; 0,655], R1 0,572; `phaseCoherence` C0 0,611, R1 0,674. 2D exploratorio ≈ 0,67. Medianas de `jpegPhaseRange` por dispositivo real entre 0,109 y 0,377: la variación entre dispositivos es mayor que la diferencia real–IA. SD2 invertido en C0 (AUC 0,373 [0,307; 0,441]), igual que en run1 (0,225) y run2-movil (0,28). Causa desconocida.

**Lectura (no prueba):** v1 no separa real/IA de forma robusta; la señal depende del dispositivo y del formato. Solo 5 fotos por dispositivo.

## 11c. Controles post hoc sobre SD2 (exploratorio; predicciones fijadas antes de ejecutar)

Observación previa (`v1-pipeline-probe-sd2.py`, run3): `jpegPhaseRange` C0 mediana 0,164 en SD2 frente a 0,282 en SDXL; mismos tamaños y recorte; rho con B/px −0,32 en SD2. Script: `bin/v1-pipeline-probe-sd2-controls.py` (A: B/px igualado por quintiles; B: OLS con log(B/px), MP y aspecto, IC bootstrap; C: pares con el mismo nombre de archivo, Wilcoxon). Sin nuevas corridas.

Predicciones, en función de la hipótesis:
- **H-textura** (SD2 es más suave/texturizado distinto; el desfase lo explica el contenido o B/px): tras A y B el efecto se reduce a la mitad o menos y su IC95 incluye 0; en C la diferencia pareada es ≈0.
- **H-propio** (propiedad de SD2 independiente de contenido y B/px): A y B mantienen el efecto (AUC SD2>SDXL ≤ 0,40; efecto OLS < 0 con IC95 que excluye 0), y C muestra SD2 menor en ≥ 65 % de los pares con p < 0,01.
- Resultado intermedio: se declara indeterminado; no se fuerza una conclusión.

Limitación: los datos son observacionales; un resultado compatible con H-propio no identifica la causa. Una intervención (reprocesar SD2) requeriría una corrida nueva y un preregistro aparte.

**Resultado 11c (run3, 905 registros) y cierre.** A) B/px igualado (n=41 por familia): AUC SD2>SDXL 0,205 [0,106; 0,312]. B) OLS: efecto SD2 −0,063 [−0,104; −0,015] (crudo ≈ −0,118). C) 16 pares con el mismo nombre: SD2 menor en 16/16, mediana −0,099, Wilcoxon p=3·10⁻⁵. Se cumple H-propio. **Cierre:** propiedad reproducible de SD2 (tres corpus), causa sin identificar; no se hace intervención (sin valor para el objetivo: SD2 queda por debajo de los reales, no sirve como detector).

## 12. Pendiente antes de congelar y ejecutar

- [ ] `node l1-manisha2025/bin/v1-pipeline-probe.mjs --inventory` y revisar elegibilidad por familia (sin extraer features).
- [ ] Revisar el borrador con el autor.
- [ ] Congelar: SHA-256 del documento, commit y etiqueta `prereg-v1probe-v1` **antes** de `--run`.
- [ ] Ejecución en 2 shards, y después `v1-pipeline-probe-analyze.py`.
