# Preregistro L1: huella global de banda media (Manisha 2025) + Mallet 2025

Estado: **BORRADOR, no congelado**. Se congela cuando se completen los puntos de la sección 12 y se etiquete el commit.
Diseño de particiones (LOGO + holdout por escena, sección 6): propuesto por el colaborador del autor en su informe y adoptado aquí.
**`progan_val.zip` no sirve para H1:** contiene un solo generador (ProGAN), por lo que no admite leave-one-generator-out. Solo se usa en el gate de replicación y en el diagnóstico de sesgo de formato.
Aislamiento: esta línea no modifica Detector v1, el registro de las 81 K, ni ningún código ni resultado existente. Vive en el directorio `l1-manisha2025/` de la misma repo `dragon3-detector-research`, sin importar módulos de `bin/` ni `src/`.

## 1. Pregunta

¿Discrimina la huella estocástica global de banda media (Manisha, Li, Kotegar, IEEE Access 13:29660, 2025, DOI 10.1109/access.2025.3537558) imágenes generadas de reales en generadores no vistos, cuando el procesado de ambas clases está igualado? ¿Añade señal la correlación de ruido entre canales (Mallet et al., AMLDS 2025, DOI 10.1109/amlds63918.2025.11159445)?

## 2. Hipótesis (fijadas antes de ver datos)

- **H1 (primaria):** la AUC macro leave-one-generator-out (LOGO) del método L1 con procesado igualado es ≥ 0,75.
- **H0:** AUC macro ≤ 0,65. Entre 0,65 y 0,75 se considera **no concluyente** (ver sección 9).
- **H2 (secundaria):** añadir las features de Mallet a L1 mejora la AUC macro LOGO en ≥ 0,03, con IC95% pareado que excluye 0.
- **H3 (secundaria):** la AUC macro bajo cada degradación de la sección 7 se mantiene ≥ 0,65.

Hay un único criterio primario (H1). H2 y H3 son secundarias y se informan sin corregir por multiplicidad.

## 3. Datos

### 3.1 Benchmark público (fase principal)
Datasets localizados (sin descargar ni inspeccionar aún):
- **DF³:** acceso por formulario (https://docs.google.com/forms/d/1STdUMSbrG-f9lWcgSEpZpi13ntg-aznPJqUFGdeTP6w/viewform). **Estado: pendiente de aprobación.** Sin acceso, la fase 1 no puede empezar.
- **CNN-aug / CNNDetection:** `progan_val.zip` (https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip). Es la partición de validación de **un solo generador (ProGAN)**. No es el test multi-generador del paper original.

Consecuencias, fijadas de antemano:
- `progan_val` **no permite LOGO** (un único generador). Solo sirve para el gate de replicación (sección 5) y para comprobar sesgos de formato. **No cuenta para H1.**
- El endpoint primario H1 (LOGO) requiere varios generadores. Fuente provisional: `CNN_synth_testset.zip` de CNNDetection (13 algoritmos CNN, carpetas `0_real` y `1_fake`; ProGAN, StyleGAN, StyleGAN2 y CycleGAN tienen varias clases en subcarpetas). Enlace del script oficial `dataset/test/download_testset.sh`: https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip (tomado del repo oficial; descarga aún no probada). Alternativa: DF³.
- Si ese enlace falla y DF³ no llega, el preregistro queda **en pausa esperando DF³**. No se sustituye por COCO.
- Limitación: son generadores CNN/GAN antiguos, no de difusión; un resultado positivo no se extrapola a SDXL, FLUX, etc.
- Riesgo conocido a verificar: en CNNDetection las reales (LSUN) podrían estar en JPEG y las IA en PNG. Si se confirma, el Pipeline de Procesado Igualado (3.3) es obligatorio y se informa el resultado sin igualar como evidencia del sesgo.

Se elige un conjunto multi-generador como principal y el otro solo como validación cruzada de dataset. Antes de congelar hay que verificar y anotar aquí:
- generadores incluidos y versión de cada uno;
- formato, resolución y compresión de reales e IA por separado;
- licencia, procedencia de las reales y existencia de identificador de escena o categoría;
- disponibilidad del código oficial de Manisha 2025.

Si reales e IA difieren en formato, tamaño o calidad JPEG, se aplica el procesado igualado de 3.3 y se registra la distribución de esos atributos antes y después.

### 3.2 Corpus propio (fase final, un solo uso)
El corpus actual de COCO y los 7 generadores del repositorio **no se usa para entrenar, calibrar ni elegir hiperparámetros**. Podrá usarse como evaluación externa, una única vez, solo si la fase 3 se supera (sección 8). Las reales de COCO ya pasaron por JPEG y resize de Flickr, por lo que su resultado se interpreta con la cautela de la sección 10.

### 3.3 Pipeline de Procesado Igualado (obligatorio)
Todas las imágenes, reales e IA, de **todos** los datasets y fases, pasan por la **misma** cadena fija antes de extraer cualquier feature:
1. **mismo decodificador** (una única biblioteca y versión, registrada) y conversión a RGB 8 bit;
2. recorte central a tamaño fijo S (valor fijado en la sección 12), sin reescalar si ambos lados ≥ S; las imágenes menores se excluyen y se cuentan por clase y generador;
3. **mismo formato de guardado: JPEG calidad 90** (Q0 = 90), 4:2:0, mismo codificador y versión registrados. Las imágenes ya en JPEG se recomprimen igual; el efecto de doble compresión se mide en el control de la sección 7.

Esto elimina el formato, el historial de compresión y el tamaño como atajos. Se informa también el rendimiento **sin** igualar, solo como diagnóstico y nunca para decidir.

## 4. Métodos

### 4.1 L1: replicación de Manisha 2025
Se replica el extractor y el clasificador descritos en el paper y, si existe, el código oficial. Toda desviación respecto al paper se anota con motivo en el registro de desviaciones (sección 11).

### 4.2 Mallet 2025 (complementaria)
Features de correlación espacial del ruido entre canales, según el paper. Se evalúan: (a) solas, (b) concatenadas con L1. El clasificador de (b) es el mismo que el de L1.

### 4.3 Baselines obligatorios
- B0: clasificador trivial con tamaño original, calidad JPEG estimada y metadatos disponibles, **antes** del procesado igualado.
- B1: Detector v1 (`jpegPhaseRange`, `phaseCoherence`) usado como caja negra, sin reentrenarlo ni tocar su snapshot.
- B2: control de contenido: el mismo extractor L1 clasificando categoría semántica, si el benchmark la tiene.

## 5. Gate de replicación (fase 0)

Antes de cualquier experimento propio, se reproduce un resultado comparable de la tabla principal de Manisha 2025 en el benchmark elegido, con el protocolo del paper. Tolerancia: la AUC o accuracy replicada difiere ≤ 0,05 de la publicada. Si no se alcanza, se documenta y **se detiene la línea** (criterio C1 de la sección 9).

## 6. Particiones

- **LOGO:** con G generadores, se entrena con G−1 y se prueba con el excluido, rotando. Las reales se reparten por escena (6).
- **Por escena:** reales y IA se agrupan por escena o categoría. Ninguna escena aparece en entrenamiento y prueba a la vez. Si el benchmark no da escena, se usa un proxy fijado antes de ver resultados (por ejemplo, el clúster de un embedding de imagen preentrenado fijo) y se declara como limitación.
- **Modelos mantenidos fuera siempre:** GPT-4o y Nano-Banana no se usan en entrenamiento ni en validación de ninguna fase.
- **Umbral:** se fija solo en el conjunto de validación del fold de entrenamiento, nunca en el de prueba. El punto de operación es FPR = 5 % en validación.
- Semilla global y semillas por fold registradas.

## 7. Controles y degradaciones

Cada degradación se aplica a prueba, con el modelo entrenado sin degradar, y también a entrenamiento y prueba juntos como variante.

| Control | Valores |
|---|---|
| Re-JPEG | calidad 95, 90, 75, 60 |
| Resize | 0,5×, 0,75×, 1,5× (Lanczos) |
| Cadena tipo red social | resize + JPEG q70, una configuración fija |
| Etiquetas permutadas | 100 permutaciones; AUC esperada ≈ 0,5 |
| Real vs. real de dos fuentes | mide cuánta señal es de fuente |
| Control de escena | B2 |

## 8. Fases y orden de análisis (no se reordenan)

1. **Fase 0:** gate de replicación (5).
2. **Fase 1:** LOGO en el benchmark principal con procesado igualado. Endpoint primario H1.
3. **Fase 2:** degradaciones y controles (7). H3.
4. **Fase 3:** Mallet solo y combinado. H2.
5. **Fase 4 (solo si H1 se cumple):** validación cruzada en el segundo benchmark y, después, **una evaluación única** en el corpus propio con GPT-4o y Nano-Banana como generadores no vistos. No hay segunda oportunidad ni reajuste tras verla.

## 9. Métricas y criterios de decisión

Métricas: AUC con IC95% por bootstrap **agrupado por escena** (≥ 2000 réplicas); TPR a FPR 1 % y 5 % con Clopper-Pearson; degradación de AUC por control; calibración del umbral entre dominios.

Reglas, fijadas de antemano:
- **Éxito (H1):** AUC macro LOGO ≥ 0,75 **y** límite inferior del IC95% > 0,65 **y** ningún generador con AUC < 0,60.
- **Fracaso (cierre):** se cumple cualquiera de:
  - **C1:** falla el gate de replicación (5).
  - **C2:** AUC macro LOGO ≤ 0,65, o límite superior del IC95% ≤ 0,70.
  - **C3:** la AUC con procesado igualado cae > 0,15 respecto a sin igualar, lo que indica que la señal es de historial de compresión.
  - **C4:** el control real-vs-real alcanza AUC > 0,65.
  - **C5:** L1 no supera a B0 en ≥ 0,05 AUC.
- **No concluyente:** AUC macro entre 0,65 y 0,75 sin cumplir C1–C5. Se informa tal cual, y no se pasa a la fase 4 sin ampliar datos con un nuevo preregistro.
- **Mallet (H2):** se incorpora solo si ΔAUC ≥ 0,03 con IC95% pareado que excluye 0. Si no, se descarta y se informa como resultado negativo.

Si se cierra por C1–C5, se publica el resultado negativo con el mismo detalle que uno positivo.

## 10. Limitaciones declaradas de antemano

- Si el benchmark no tiene identificador de escena, el holdout por escena es aproximado.
- Un detector puede aprender rasgos del dataset aunque se iguale el procesado; la validación cruzada de fase 4 mitiga, no elimina.
- La fase 4 con COCO hereda su historial de compresión; un resultado positivo ahí no prueba origen generativo.
- Nano-Banana y GPT-4o son cajas negras con pipeline de salida desconocido.
- Los resultados no son una certificación forense ni validación comercial.

## 11. Registro de desviaciones

Cualquier cambio posterior al congelado se anota con fecha, motivo y efecto previsto, en un apéndice. Los análisis no previstos se etiquetan **exploratorios** y no entran en las decisiones de la sección 9.

**D1 (2026-11-11, antes de congelar y antes de ver datos reales).** Partición: un único corte entrenar con 10 generadores y probar con 3 (`stylegan2`, `gaugan`, `whichfaceisreal`), en lugar de LOGO con rotación completa (sección 6). Elegidos por el autor antes de ejecutar. Efecto: la AUC depende de esos 3 generadores y no es una estimación LOGO; se declara así. Parámetros aprobados: S=256, K=32, JPEG Q90 4:2:0, banda [0,02, 0,35).
**D2.** Método: el script `l1-manisha2025/bin/l1-proxy-logo.py` es un **proxy genérico** (espectro radial de potencia), no la réplica de Manisha 2025. Leído el paper, el método real usa una ResNet101 preentrenada (identificación de cámara) + SVM RBF, no un espectro FFT/DCT; no hay pesos ni código públicos (ver `docs/EMAIL_AUTORES_MANISHA.md`). Sus resultados son **exploratorios** y no cuentan para el gate (sección 5) ni para H1.
**D3.** Escena: el dataset CNN_synth_testset no trae identificador de escena; el holdout por escena **no se hace** en esta ejecución. Limitación declarada.
**D4.** Control de etiquetas permutadas: 100 permutaciones, se informa media y SE (una sola permutación no es informativa).

**D5.** Tras la inspección de sesgo de formato (6/13 alertas), se excluyen `whichfaceisreal` (reales/falsas JPEG con compresión distinta: blockiness 1.134 vs 0.999, bytes/píxel >50%), `san` (481x321 vs 480x320, blockiness fake 1.078) y `deepfake` (tamaños distintos 256x278 vs 256x263). Quedan 10 generadores: 7 de entrenamiento y 3 de prueba (`stylegan2`, `gaugan`, `crn`). Se corrige la regla de tamaños (similitud de Jaccard de los 3 tamaños más frecuentes, alerta si < 0.5), que daba falsos positivos en `stylegan2` y `seeingdark`. Q90 no elimina una compresión previa más fuerte (hallazgo T6), por lo que la exclusión es necesaria. Encuadre: **exploratorio L1-proxy**, no es réplica de Manisha 2025 ni valida H1; la validación de H1 será con el corpus de difusión propio con reales igualadas (opción C). DF³ y la réplica real de Manisha siguen pendientes.
**D6 (2026-11-11, tras ver los resultados del run1).** (a) El run1 se cierra por C2 como resultado negativo exploratorio (sección 14). (b) El control de tamaño/bytes mostró que `crn` se separa casi perfectamente por tamaño/bytes originales (AUC 0.026, es decir 0.974 invertido) y la inspección no lo había marcado; `inspect-format-bias.py` incorpora ahora el AUC univariante por clase de bytes, bytes/píxel, ancho, alto, área y blockiness, con alerta si |AUC−0.5| > 0.15 (muestra por defecto 200). (c) Se añade `bin/l1-mallet.py` (extractor de correlación de ruido de Mallet 2025). Desviaciones propias de ese script: T no está especificado en el paper y se fija en 128; clasificador logístico binario en lugar del esquema de 3 pasos; 20 permutaciones; Pipeline de Procesado Igualado Q90. (d) No se reajustan umbrales, features ni generadores tras ver el resultado.

**D7 (2026-11-11, tras la inspección 2).** CNN_synth_testset se cierra definitivamente. Tras añadir el AUC univariante, solo `gaugan`, `progan`, `stylegan` y `stylegan2` quedan limpios; `seeingdark` (blockiness AUC 0.91), `crn`, `imle` y `stargan` (bytes) están contaminados. No se hace run2. El run1 mantiene su resultado negativo (C2). La línea continúa con el corpus de difusión propio en `PREREGISTRO_L2_MALLET.md`.

**Nota sobre el autotest sintético:** una primera versión falló T1 y T2b por defectos del propio test (frecuencia del patrón distinta por generador, y una única permutación); se corrigieron y no se modificaron los umbrales. Hallazgo informativo T6: con reales JPEG q75 y falsas PNG sin señal real, la AUC espuria fue 0,95 sin igualar y 0,98 con Q90, es decir, **Q90 no borra la huella de una compresión previa más fuerte**.

## 12. Pendiente antes de congelar (no inventado, hay que verificar)

- [ ] Leer el paper completo de Manisha 2025 y extraer: definición exacta del extractor, banda, clasificador, hiperparámetros, tabla de resultados a replicar.
- [ ] Leer el paper completo de Mallet 2025 y extraer la definición de las features.
- [ ] Confirmar disponibilidad y licencia de código y datos de ambos.
- [ ] Solicitar DF³ por formulario y anotar fecha de solicitud y de respuesta.
- [ ] Descargar `progan_val.zip`, registrar SHA-256, nº de imágenes, formato y compresión de reales vs IA.
- [ ] Localizar el test multi-generador de CNNDetection (13 generadores) y verificar su disponibilidad actual.
- [ ] Elegir el conjunto multi-generador principal (3.1); `progan_val` no puede serlo.
- [ ] Aclarar el alcance de banda: el paper de 2022 habla de bandas baja y media; este preregistro usa la banda exacta que defina Manisha 2025, no la [0,02, 0,35) de v1.
- [ ] Decidir si el trabajo vive en esta repo o en `dragon3-image-authenticity` (antes se acordó: misma repo, directorio `l1-manisha2025/`).
- [ ] Fijar S y el proxy de escena si hace falta (Q0 = 90 ya decidido).
- [ ] Fijar clasificador y búsqueda de hiperparámetros (solo en validación).
- [ ] Registrar versiones de Python, bibliotecas, codificador JPEG y hardware.
- [ ] Congelar: hash SHA-256 de este documento, commit y etiqueta `prereg-l1-v1` antes de ejecutar la fase 1.

## 13. Reproducibilidad y aislamiento

Misma repo que Detector v1 (`dragon3-detector-research`), rama `l1-manisha2025`, directorio `/home/gustavo/dragon3-detector-research/l1-manisha2025/` (decisión del autor), con entorno propio de versiones fijadas, semillas registradas, predicciones por imagen y hashes de resultados. No se modifica ningún archivo fuera de ese directorio; la integridad de v1 se comprueba con `python3 bin/freeze-detector-v1.py --verify` antes y después de cada fase. El detector v1 solo se invoca como baseline B1 en modo lectura. El congelado usa un commit y una etiqueta `prereg-l1-v1` en esa misma repo.

## 14. Resultados

### 14.1 Run1 exploratorio L1-proxy (2026-11-11; commit del script `8a31a5e`, SHA-256 `123cc8e3…6101`)

Datos: CNN_synth_testset, 10 generadores (excluidos `whichfaceisreal`, `san`, `deepfake`); entrenamiento con `biggan`, `cyclegan`, `imle`, `progan`, `seeingdark`, `stargan`, `stylegan`; prueba con `crn`, `gaugan`, `stylegan2`. 43 746 imágenes de entrenamiento y 38 740 de prueba. Semilla 20261111.

| Medida | Resultado |
|---|---|
| AUC macro igualado (Q90) | 0.5424 (IC95 0.537–0.548, bootstrap por imagen) |
| AUC macro sin igualar (diagnóstico) | 0.5354 |
| `crn` / `gaugan` / `stylegan2` | 0.647 / 0.498 / 0.483 |
| Etiquetas permutadas (100) | media 0.499, SE 0.0026 |
| TPR a FPR de validación 5 % | 8.7 % / 2.2 % / 1.7 % (FPR real 1.8 % / 2.9 % / 3.9 %) |
| Control tamaño/bytes (AUC por generador) | `crn` 0.026, `gaugan` 0.471, `stylegan2` 0.581 |

Lectura:
- **C2 se cumple** (AUC macro ≤ 0.65, y el criterio de éxito de H1 no se alcanza): resultado negativo para este proxy. Es **exploratorio** y no cuenta para el gate ni para H1.
- C3 no se activa (sin caída al igualar); la señal débil no procede de la compresión. El control de permutaciones indica que el pipeline no filtra información.
- `crn` es la única señal (0.647) y está contaminada: tamaño/bytes lo separan casi por completo. El macro del control de tamaño (0.359) no es interpretable por la inversión de signo; leer |AUC−0.5|.
- El IC95 subestima la incertidumbre porque la varianza dominante es entre generadores (n = 3 de prueba, un único corte).
- No se midió la AUC dentro de distribución, por lo que no se distingue entre "features inútiles" y "features específicas de generador".
- Esto no refuta a Manisha 2025 (no se replicó) ni dice nada sobre difusión.

### 14.2 Pasos siguientes (no ejecutados)

1. Rehacer la inspección con la versión nueva de `inspect-format-bias.py` (revisar `crn`, `imle`, `gaugan`).
2. Opción C: corpus de difusión propio con reales igualadas, evaluado con `l1-mallet.py` (`--root`, `--test-gens`, `--exclude`) y registrado en un nuevo preregistro antes de ejecutar.
3. Pedir pesos y código a los autores de Manisha 2025.
4. Esperar la respuesta del formulario de DF³.
