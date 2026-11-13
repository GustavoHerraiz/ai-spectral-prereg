# Preregistro Línea 2 (Opción C): correlación de ruido (Mallet 2025) sobre corpus de difusión propio

**Estado: BORRADOR, no congelado. No hay código ni resultados.** Autor: Gustavo Herraiz. Fecha: 2026-11-11.
Misma repo y rama (`l1-manisha2025`), directorio `l1-manisha2025/`. No se toca v1, `v3/` ni las 81 K. CNN_synth_testset queda cerrado (ver D7 en `PREREGISTRO.md`).

## 1. Pregunta e hipótesis (reencuadre: extensión, no réplica)

Los papers (Mallet 2025, Manisha 2025) son punto de partida. Este estudio **no replica** sus resultados; los supera en rigor (LOGO, procesado igualado, controles de sesgo, criterios de cierre previos).
- **H1-L2:** la huella de **bajas frecuencias** (propuesta propia; extractor definido y congelado antes de ejecutar, p. ej. `l1-proxy-logo.py`) separa reales de difusión de familias no vistas, tras igualar el procesado y superar los controles.
- **B1 (comparador):** Mallet 2025 (`l1-mallet.py`, T=128 fijo).
- **H2-L2:** la fusión (concatenación) de H1 y B1 mejora a cada uno por separado en ≥ 0,03 AUC macro, con IC95 pareado que excluye 0.
- **H0:** AUC macro LOGO ≤ 0,65 o explicable por tamaño/bytes/compresión.
- Sin gate de replicación. Alcance exploratorio-confirmatorio, no forense.

## 2. Datos: Synthbuster (sustituye al corpus propio, cerrado por contaminación de procesado)

Config: `config/l2-synthbuster.json`. Raíz: `/home/gustavo/dataset_test/synthbuster/`. Reales: RAISE-1k; sin JPEG ni remuestreo según la ficha de Zenodo (no verificado localmente).

| Familia (unidad de holdout) | Carpetas |
|---|---|
| dalle2, dalle3, firefly, glide, midjourney, sd2, sdxl | una cada una |
| sd1 | stable-diffusion-1-3 + stable-diffusion-1-4 |

**LOGO de 8 familias**: entrenar con 7, probar con la retenida. Las reales se reparten en 8 grupos (semilla 20261111) y las del grupo k no entran en el entrenamiento del pliegue k. Corpus propio (COCO, ai_*, móvil, GANs L1): solo prueba de humo y controles, nunca evidencia.
Crops: 512×512 central sin resize (`build-l2-corpus.py`); imágenes menores se descartan y se cuentan. Verificar tamaños con el recuento sin `--execute` antes de congelar.

### 2.1 Procedencia que falta (bloquea el congelado)
Mallet mide correlación entre píxeles vecinos, que cambia con cualquier remuestreo. Hay que verificar para Synthbuster (resoluciones por generador, formato de RAISE-1k, recorte aplicado):
1. Resolución nativa de cada generador y cómo se llegó a 512×512 (reescalado, recorte, nativo).
2. Formato y tamaño de las reales RAISE-1k tal como se descargan (TIFF/PNG/JPEG, resolución). Las reales llevan **historial JPEG y posible remuestreo**; las fakes, probablemente no. Es el mayor riesgo del diseño (hallazgo T6 de L1: el Q90 no borra una compresión previa más fuerte).
4. Prompts, semillas y si las fakes se generaron con prompts de las reales COCO (contenido emparejado) o no.

## 3. Criterio de inclusión (fijado antes de mirar datos)

Se calcula **solo con bytes, sin clasificador ni etiquetas de entrenamiento**, por familia f:
- AUC univariante de `bytes` del PNG original, reales vs fakes de f (para nano_banana, ambas carpetas juntas).
- Misma medida tras el procesado igualado (bytes del JPEG Q90).

**Una familia se incluye si ambos AUC están en [0,40; 0,60]** (versión bilateral del «≤ 0,6» pedido: en L1, `crn` e `imle` dieron AUC muy bajos, que también es separación). Con n = 50 el SE es ≈ 0,058, así que ≈ 9 % de familias limpias se excluirán por azar; se acepta. Familias excluidas: se informan con su AUC y no se rescatan después. **Si quedan < 5 familias, no se hace LOGO y se cierra la línea como no concluyente.**

Declaración honesta: la regla no está ciega al corpus (ya existe), pero sí al resultado del clasificador. Si el autor ya vio AUC de bytes de estas carpetas, debe anotarlo aquí antes de congelar: ______.

## 4. Pipeline de Procesado Igualado

Reutiliza `l1-proxy-logo.py` (`center_crop`, `equalize`): RGB 8 bit → recorte central S×S **sin reescalar** → JPEG Q90 4:2:0 en memoria → decodificar. S = 256 por defecto, y 512 como análisis de sensibilidad. Brazos:
- **Primario:** Q90 igualado.
- **Sensibilidad A:** sin igualar.
- **Sensibilidad B:** doble cadena idéntica Q75 → Q90 para reales y fakes (atenúa, no elimina, el historial de las reales).

## 5. Feature principal (Mallet 2025, sección II.B)

Filtro Laplaciano L4 sobre Y, Cb, Cr → bloques 8×8 vectorizados → unión de los T bloques de menor varianza y los T de menor media → matriz de correlación 64×64 → triángulo inferior: 3 × 2016 = 6048 features. `l1-mallet.py` ya lo implementa (autotest 7/7). **T no lo especifica el paper: T = 128 fijo, sin ajuste.** Clasificador: regresión logística L2 sobre features estandarizadas. Con ~350 imágenes y 6048 features hay riesgo de sobreajuste: C se elige solo por validación cruzada interna, en {0,001; 0,01; 0,1; 1}, **dentro del entrenamiento**, nunca con la familia de prueba. Se declaran como desviaciones: T, clasificador binario en vez del esquema de 3 pasos, y que el paper no pasa por Q90 ni 4:2:0 (el submuestreo de croma afecta a Cb/Cr; se informa también el AUC solo con Y).

## 6. Holdout LOGO por familia

8 pliegues. En cada uno: entrenar con 7 familias, probar con la retenida. **Las reales también se particionan**: se reparten al azar (semilla fija, fijada aquí: 20261111) en 8 grupos; las reales del grupo k no entran en el entrenamiento del pliegue k. AUC por familia: 50 fakes de f contra **todas** las reales puntuadas fuera de muestra (cada una por el modelo de su pliego). Limitación: esas puntuaciones vienen de 7 modelos distintos; se informan también las AUC por pliego (con ≈ 7 reales) como diagnóstico, no como decisión. No hay escena: el holdout por escena **no se hace** (limitación). Limitación adicional: familias que comparten decodificador (entiendo que PixArt-Σ usa el VAE de SDXL; verificar) no son independientes.

## 7. Igualar la complejidad (bytes) entre reales RAISE y fakes PNG

**Propuesta: no igualar por transformación; controlar por selección y por diseño.** Cualquier transformación que iguale bytes (ruido, desenfoque, recompresión fuerte) actúa sobre el mismo ruido que Mallet mide y destruye o inventa señal.
1. **Criterio de inclusión** (sección 3).
2. **Análisis emparejado (sensibilidad):** emparejar cada fake con la real de bytes-Q90 más cercano (con calibre; sin reemplazo cuando se pueda) y repetir el LOGO solo con pares emparejados. Se informa cuántos pares se pierden.
3. **Estratificación:** AUC dentro de terciles de complejidad.
4. **Segunda fuente de reales (recomendada):** fotos de cámara sin recompresión, si existen en `prnu_registry` (no verificado), pasadas por el mismo pipeline. Dos controles: (i) AUC reales-COCO vs reales-cámara; si es > 0,65, las features codifican la procedencia de las reales y el resultado no vale; (ii) entrenar con COCO y probar contra reales de cámara.
5. **Baseline B0:** regresión logística solo con bytes/bytes·píxel (y recorte). L2 debe superar a B0 en ≥ 0,05 AUC.

**Si no se puede igualar sin destruir la señal, se justifica así en el informe:** la confusión de complejidad e historial de compresión **solo puede inflar un resultado positivo, no producir uno negativo**. Por tanto, un positivo se redacta como «separa estas reales COCO de estos generadores», condicionado a los controles 2–5, y no como «detecta origen generativo». Un negativo sí es robusto y se publica con el mismo detalle.

## 8. Métricas y criterios de decisión (fijados de antemano)

Métricas: AUC por familia y macro; IC95 por bootstrap (≥ 2000 réplicas) remuestreando imágenes dentro de cada familia; TPR a FPR 5 % con el umbral fijado en validación interna; permutaciones de etiquetas (100, media y SE); control reales-vs-reales (reparto aleatorio de reales con etiquetas ficticias).
- **Éxito:** AUC macro ≥ 0,75 **y** límite inferior IC95 > 0,65 **y** ninguna familia < 0,60 **y** ninguno de C1–C4 activado.
- **C1:** AUC macro ≤ 0,65 o límite superior ≤ 0,70.
- **C2:** AUC igualado cae > 0,15 respecto al brazo sin igualar.
- **C3:** control reales-vs-reales o RAISE-vs-cámara (si hay segunda fuente) > 0,65.
- **C4:** no supera a B0 en ≥ 0,05.
- **No concluyente:** AUC macro entre 0,65 y 0,75 sin C1–C4, o < 5 familias incluidas.
- **Advertencia de potencia:** con 50 imágenes el IC por familia es ancho; un «ninguna familia < 0,60» puede fallar por azar. Se informa tal cual, sin repetir el análisis para forzar el éxito.

## 9. Análisis aparte
- Edición de nano_banana: sin etiqueta fake «pura», se reporta como análisis exploratorio.
- Robustez a JPEG Q90/75/60 y reescalado (como en Mallet), solo informativo.

## 10. Limitaciones declaradas
Corpus pequeño; una sola fuente de reales con historial JPEG; sin identificador de escena; familias con decodificador compartido; procedencia de la resolución 512 sin verificar; Mallet no está replicado (datos y clasificador distintos), y no hay gate de replicación posible porque su dataset no es público aquí.

## 11. Registro de desviaciones
- L2-D1: `bin/l2-mallet.py` implementa Mallet con T=128 bloques (el paper no fija T) y regresión logística binaria L2 (el paper usa 3 pasos); C se elige por CV interna de 3 pliegues.
- L2-D2: se aplica Q90/4:2:0 a ambas clases (el paper no iguala); afecta a Cb/Cr.
- L2-D3: AUC por familia = fakes de la familia contra todos los reales fuera de pliegue (8 grupos de reales, semilla 20261111); los puntajes provienen de 8 modelos distintos.
- L2-D4: la puerta de bytes (PNG y JPEG Q90 del recorte, AUC en [0,40; 0,60]) PARA el script por defecto (código 2) sin extraer features; `--allow-exclude` aplica la regla preregistrada (excluir y seguir si quedan ≥5 familias).
- L2-D5: H1 (huella propia de baja frecuencia) aún no tiene extractor congelado; esta ejecución solo cubre B1 (Mallet) y controles.

## 12. Pendiente antes de congelar
- [ ] Descargar Synthbuster y real_RAISE_1k; ejecutar `build-l2-corpus.py` sin `--execute` y revisar el recuento (≥512 en ambos lados).
- [ ] Ejecutar `inspect-format-bias.py` y el criterio bilateral de bytes [0,40; 0,60] ANTES de cualquier clasificador.
- [ ] Respuestas de 2.1 (procedencia, prompts, resolución nativa).
- [ ] Decidir ampliar el corpus (≥ 200 reales, ≥ 100 por familia).
- [ ] Confirmar si existe una segunda fuente de reales sin recompresión.
- [ ] Declarar si se vieron antes AUC de bytes en este corpus.
- [ ] Revisar el borrador con el autor y los colaboradores.
- [ ] Congelar: SHA-256 del documento, commit y etiqueta `prereg-l2-v1` **antes** de ejecutar.
- **L2-D6** (antes de la ejecución real): dalle3 es RGBA (diagnóstico `gate_bytes.json`, 50/50 descartadas por `modo:RGBA`). Se aceptan RGBA con alfa uniformemente 255, convertidas a RGB sin pérdida. RGBA con alfa no opaco se sigue descartando.
- **L2-D7**: GLIDE (256 px) queda excluida por la regla de ≥ 512 sin reescalar. El LOGO pasa a 7 familias (`--expect-families 7`).
- **L2-D8** (limitación): las reales RAISE-1k están reducidas ≈3,9× (1258×833) y las IA son nativas; firefly es JPEG. Los resultados de L2 son exploratorios y con confusión de remuestreo.
- **L2-D9** (antes de extraer ninguna feature; motivada por el resultado de la puerta en la ejecución 1: AUC de bytes PNG 0,25–0,82, 5/7 familias fuera de [0,40, 0,60]): igualado de complejidad con `--match-bytes`. Solo usa bytes PNG, nunca features. Contenedores = deciles de los bytes PNG de los reales; se conservan los contenedores con ≥ 5 imágenes en **todas** las familias; los reales se restringen a ellos y cada familia se submuestrea para reproducir la distribución de los reales (semilla fija). La puerta [0,40, 0,60] se **recalcula después** del igualado, sobre bytes PNG y JPEG, y no se relaja. Si no hay soporte común (< 3 contenedores) o la puerta sigue fallando, se para sin extraer features. Se guarda `gate_bytes.json` con AUC antes (`families_sin_igualar`) y después, y con el detalle de `igualado`. Limitaciones: (i) igualar bytes puede eliminar parte de la señal real si esta covaría con la complejidad; (ii) restringe los reales al rango común; (iii) no corrige el remuestreo previo de las reales ni el JPEG de firefly, solo la complejidad marginal. Cualquier resultado se reporta junto al de la ejecución 1 (puerta fallida).
