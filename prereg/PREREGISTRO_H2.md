# Preregistro H2: ¿qué es el desplazamiento de sd-turbo sin cadena?

Rama `h2-sdturbo`, creada desde `h1-replicacion` (`1ac264e` → paper `9b50867`). `l1-manisha2025` y `h1-replicacion` no se tocan. Fecha: 2026-11-12. Redactado **antes** de ejecutar ningún análisis de esta rama.

## 1. Pregunta y estatuto
En §5.13 (exploratorio) se observó que, con un modelo entrenado en Synthbuster A5 frente a reales R1, las imágenes de sd-turbo **sin cadena (A1)** ya puntúan «IA» (A1 vs R1 ≈ 0,89–0,91), mientras que las 8 carpetas `ai_*` puntúan como los reales (A1 vs R1 0,19–0,78). Este preregistro intenta distinguir las causas candidatas. **H2 es una investigación de una pista post hoc: sus resultados son confirmatorios solo respecto a los criterios de abajo, y el alcance se limita a sd-turbo y a este modelo.**

Lo ya visto antes de redactar (no se vuelve a interpretar como resultado nuevo): los valores de §5.13 y la tabla de `h1-chain`. **No se ha ejecutado** ninguna de las exclusiones de familia ni ninguna regeneración descritas aquí.

## 2. Hipótesis candidatas
- **C1. Propiedad del generador:** sd-turbo (destilado, pocos pasos) deja un rasgo propio.
- **C2. Pipeline propio:** float32, attention slicing, PNG; algo de la generación local, no del modelo.
- **C3. Historial de las otras 8 carpetas** (reescalado/JPEG previos) borra el rasgo; sin generación propia no es separable.
- **C4 (nueva). Linaje Stable Diffusion del entrenamiento:** sd-turbo deriva de SD 2.1 (según su documentación pública, no verificado aquí) y Synthbuster incluye SD1, SD2 y SDXL. Si el modelo aprendió un rasgo de decodificador/arquitectura SD presente en esas familias, sd-turbo se parecería a ellas **sin que sea una propiedad de «IA»**. §5.1 ya registra una propiedad de SD2 «reproducible en tres corpus, sin causa identificada».

## 3. Etapa 0: sin generar nada (`bin/h2-sdfamily.py`, autotest 4 comprobaciones)
**Qué.** Rasgo `res_spec`. Modelo: Synthbuster A5 (con familias excluidas) frente a R1 de un corpus; prueba: AUC de sd-turbo **A1** frente a R1 del otro corpus (run3↔Dresden, media de los dos sentidos), IC95 por bootstrap sobre dispositivos (reales) e imágenes (IA), semilla 20261111. Entradas archivadas: `h1-resample-feats.jsonl`, `h1-owngen-feats.jsonl`.
**Entrenamientos.**
- X0: todas las familias (referencia; debe reproducir ≈0,9; si no, se detiene y se investiga el script).
- X1: sin SD2. X2: sin SD1 y SD2. **X3 (primario): sin SD1, SD2 y SDXL.**
- **Controles de pérdida genérica:** excluir 3 de las 4 familias no SD (dalle2, dalle3, firefly, midjourney), las 4 combinaciones. Quitan el mismo número de familias que X3 sin tocar el linaje SD.

**Criterios (fijados antes).** Umbrales de §21: «no explicada» si media ≥ 0,75 y mín. IC inf. ≥ 0,60; «explicada» si media < 0,68.
- **Primario:** X3 media < 0,68 **y** media de los 4 controles ≥ 0,75 → *explicada por linaje SD del entrenamiento* (C4).
- X3 < 0,68 pero controles también < 0,75 → *indeterminada* (pérdida genérica de datos de entrenamiento, no linaje).
- X3 ≥ 0,75 con mín. IC inf. ≥ 0,60 → *no explicada por linaje SD* (C4 descartada para este modelo).
- En otro caso, *indeterminada*.
X1 y X2 son localizadores descriptivos (qué familia pesa); no cambian el veredicto.

**Predicciones (probabilidad antes de ver datos).** X3 < 0,68 y controles ≥ 0,75 (C4): **p ≈ 0,35**. X3 ≥ 0,75 (C4 descartada): p ≈ 0,45. Resto: p ≈ 0,20.

**Decisión posterior (fijada ahora).** Si el veredicto es «explicada por linaje SD», la pista se cierra como artefacto del entrenamiento, **no se genera nada** y el manuscrito lo recoge como nota. En cualquier otro caso se pasa a la etapa 1.

## 4. Etapa 1: generación propia (solo si procede según §3; ~16 h de CPU, con `pm2 stop all`)
Mismas condiciones que §19 (H1-D3: float32, attention y VAE slicing, 512×512 PNG, manifiesto con SHA-256 y versiones), semillas y prompts **disjuntos** de los 100 anteriores (desplazamiento de semilla +200000 y 20 escenas nuevas, `SCENES2`; opciones `--steps`, `--seed-offset` y `--new-prompts` añadidas a `h1-owngen-make.py` en un commit separado, con autotest 7/7, **antes** de generar; ver erratum en §7). n = 30 por condición:
- **E1 réplica:** 2 pasos, guidance 0, ajustes idénticos al original.
- **E2a:** 1 paso. **E2b:** 4 pasos. (dosis-respuesta con los pasos).
- Estadístico: S = AUC de A1 frente a R1 (modelo Synthbuster A5/R1 completo, como X0), media run3↔Dresden.
**Criterios.**
- **Réplica (E1):** S ≥ 0,75 con mín. IC inf. ≥ 0,60 → el desplazamiento se reproduce; S < 0,68 → **no se reproduce** (el original era propio de esas semillas/prompts o de esa ejecución) y se cierra.
- **Pasos (solo si E1 se reproduce):** «depende de los pasos» si |S(1) − S(4)| ≥ 0,15 y el IC95 (bootstrap) de la diferencia excluye 0; si no, «no depende».
**Lectura prefijada.** Se reproduce y depende de los pasos → compatible con un rasgo de destilación/pocos pasos (C1), sin probarlo (un solo modelo). Se reproduce y no depende → propiedad del modelo o del pipeline sin poder separar C1 y C2. No se reproduce → C1 y C2 no sostenidas por estos datos.
**Predicciones.** E1 se reproduce: p ≈ 0,75. Depende de los pasos (dado que se reproduce): p ≈ 0,35.

## 5. Lo que H2 no puede resolver (declarado de antemano)
- **C2 (pipeline) no se aísla:** attention/VAE slicing son equivalentes matemáticos salvo redondeo y bfloat16 fue inviable (H1-D2); cambiar el pipeline no es factible en este servidor (3,8 GB de RAM). Un resultado que se reproduce solo descarta que sea fruto del azar de una tanda, no del pipeline.
- **C3 (historial de las otras carpetas) no se aísla** sin generación propia de esos modelos. **SDXL-turbo y sd15 no caben en la RAM** del servidor; un generador distinto sería otro estudio.
- Un solo modelo (sd-turbo), n = 30 por condición, mismo clasificador. No generaliza a «los generadores destilados».
- Estos resultados no modifican el veredicto de H1 («no apoyada»).

## 6. Ejecución
Etapa 0 (minutos, no toca el servidor salvo `results/h2-e0-run1/`):
`python3 l1-manisha2025/bin/h2-sdfamily.py --self-test` y luego `--resample-feats …/h1-resample-feats.jsonl --own-feats …/h1-owngen-feats.jsonl --out l1-manisha2025/results/h2-e0-run1`. Archivado con `archive-results.py` (añadir `h2-e0-run1` a `--names`). Cualquier resultado se informa.

## 7. Resultados de la etapa 0 y erratum
**Ejecución:** servidor, commit `9f8fbd5` (tag `h2-criterios`), autotest 4/4, `results/h2-e0-run1/h2-sdfamily.json`.

| entrenamiento | excluye | media | mín. IC inf. |
|---|---|---|---|
| X0 | nada | 0,899 | 0,847 |
| X1 | SD2 | 0,899 | 0,845 |
| X2 | SD1, SD2 | 0,908 | 0,856 |
| **X3 (primario)** | SD1, SD2, SDXL | **0,919** | **0,867** |
| controles (4) | 3 familias no SD | 0,884–0,904 (media 0,891) | 0,821–0,842 |

**Veredicto: «no explicada por linaje SD»** (X3 ≥ 0,75 con mín. IC inf. ≥ 0,60). X0 reproduce el ≈0,9 de §5.13. La predicción de C4 (p ≈ 0,35) **no se cumplió**; el desplazamiento de sd-turbo sin cadena no depende de que el entrenamiento contenga familias SD. Alcance: este modelo, este rasgo, un solo generador. Quedan C1, C2 y C3. Por la regla de §3 se pasa a la etapa 1.

**Erratum (§4).** El preregistro prometía prompts «disjuntos» de los originales, pero el espacio de `prompt_of` solo tiene 40 prompts, todos usados en los 100 originales. Se resuelve con 20 escenas nuevas (`SCENES2`, 40 prompts nuevos) sin tocar `prompt_of` por defecto. No cambia criterios ni predicciones. La etapa 1 usa las mismas semillas y prompts en las tres condiciones (1, 2 y 4 pasos), por lo que están emparejadas.

## 8. Ejecución de la etapa 1
Evaluación: `bin/h2-steps.py` (autotest 6/6; réplica y pasos con los criterios de §4; diferencia S(1) − S(4) con bootstrap emparejado por semilla y por dispositivo). Generación: `h1-owngen-make.py --only sdturbo --n-sdturbo 30 --seed-offset 200000 --new-prompts` con `--steps 2`, `--steps 1` y `--steps 4`, cada uno en su carpeta (`/home/gustavo/dataset_test/owngen_h2/{s2,s1,s4}`), una tras otra, con `pm2 stop all` y reanudable. Los PNG se empaquetan en un `.tar` con SHA-256 como en H1. El análisis no se ejecuta hasta terminar las tres tandas.

## 9. Resultados de la etapa 1
**Ejecución:** servidor, commit `fc5c6f5`, autotest 6/6, `results/h2-e1-run1/h2-steps.json`. Generación: 30 imágenes por condición, semillas `20261112 + 200000 + i`, `SCENES2`, float32. `.tar` de las 90 imágenes: SHA-256 `e49ffbb81504f233666563a0aad2bc7b857349b9df8885e3b370c4cde0ab53d8` (copia en el PC del autor).

| condición | n | S | mín. IC inf. | veredicto base |
|---|---|---|---|---|
| E2a, 1 paso | 30 | 0,375 | 0,230 | artefacto de corpus |
| **E1, 2 pasos (réplica)** | 30 | **0,816** | 0,704 | aguanta |
| E2b, 4 pasos | 30 | 0,572 | 0,367 | artefacto de corpus |

S(1) − S(4) = −0,197, IC95 [−0,311; −0,086].

**Veredicto según §4: «se reproduce» y «depende de los pasos».** Predicciones: réplica (p ≈ 0,75) cumplida; dependencia de los pasos (p ≈ 0,35) cumplida.

**Matiz que corrige la lectura prefijada.** La lectura de §4 («compatible con un rasgo de destilación/pocos pasos, C1») asumía un efecto monótono. El patrón observado **no es monótono**: máximo en 2 pasos y descenso a ambos lados; con 1 paso S < 0,5 (esas imágenes puntúan más «reales» que los reales). Un rasgo de «pocos pasos» no debería desaparecer con 1 paso, de modo que C1 **no queda apoyada** por este patrón. La réplica es además más débil que el original (0,816 frente a 0,899). Con un solo modelo, n = 30 y una tanda, no se separan C1 y C2. Lo que se confirma es que el desplazamiento existe, se repite con semillas y prompts nuevos, y es específico de la configuración.

## 10. Etapa 2 (exploratoria, preregistrada antes de ejecutar): ¿lo explica un descriptor simple?
**Hipótesis post hoc (origen: el patrón no monótono de §9):** `res_spec` responde a la nitidez o contenido de alta frecuencia de la imagen, no a su origen; cada configuración de pasos produce imágenes con distinta textura.
**Qué.** `bin/h2-descriptors.py` (autotest 4/4, ruta completa probada con datos sintéticos). Sobre las 90 imágenes de §9, en su versión A1 (recorte alineado + JPEG q92): puntaje = decision_function del modelo Synthbuster A5 vs todos los R1; descriptores `d_lap` (log10 de la varianza del laplaciano del gris) y `d_hf` (fracción de energía espectral con radio > 0,25 ciclos/px, ventana de Hann). Ambos se fijan aquí; no se añadirán otros.
**Estadístico.** Spearman agrupado (90 imágenes) puntaje~descriptor, IC95 por bootstrap de semillas (las tres condiciones de una semilla se remuestrean juntas).
**Criterios (fijados antes).**
- «El descriptor X sigue al puntaje»: |rho| ≥ 0,6 y el IC95 queda a ≥ 0,4 de cero, con el mismo signo.
- «Ningún descriptor sigue al puntaje»: |rho| < 0,3 en ambos.
- En otro caso, indeterminada.
**Predicciones.** Algún descriptor sigue al puntaje: p ≈ 0,45; ninguno: p ≈ 0,30; indeterminada: p ≈ 0,25.
**Lectura prefijada.** Si un descriptor lo sigue, el puntaje de sd-turbo refleja una propiedad de textura y no una huella de generación (coherente con H1), **sin probar causalidad** (correlación, un modelo, 90 imágenes). Si ninguno, la pista sigue sin explicación y se cierra como no resuelta. No se compara con reales: los descriptores solo se evalúan entre las imágenes propias.
**Límites.** Dos descriptores elegidos por mí; una correlación agrupada puede estar dominada por la diferencia entre condiciones (se informan también las medianas por condición, descriptivas). La etapa 2 no genera imágenes y no puede abrir más etapas: tras ella H2 se cierra.
**Ejecución.** `python3 l1-manisha2025/bin/h2-descriptors.py --s1-dir $D/s1 --s2-dir $D/s2 --s4-dir $D/s4 --resample-feats l1-manisha2025/results-archive/h1-resample-run1/h1-resample-feats.jsonl --out l1-manisha2025/results/h2-e2-run1`.

## 11. Resultados de la etapa 2 y cierre de H2
**Ejecución:** servidor, commit `ba265f1`, autotest 4/4, `results/h2-e2-run1/h2-descriptors.json`, 90 imágenes.

| condición | puntaje mediano | d_lap mediano | d_hf mediano |
|---|---|---|---|
| 1 paso | −2,492 | 2,990 | 0,0263 |
| 2 pasos | 0,496 | 3,101 | 0,0202 |
| 4 pasos | −1,328 | 3,088 | 0,0077 |

Spearman agrupado: puntaje~`d_lap` = 0,094 (IC95 [−0,107; 0,285]); puntaje~`d_hf` = 0,163 (IC95 [−0,015; 0,320]).

**Veredicto: «ningún descriptor sigue al puntaje»** (|rho| < 0,3 en ambos). La predicción «ninguno» (p ≈ 0,30) se cumplió; la hipótesis de nitidez/alta frecuencia **queda refutada** para estos dos descriptores. Los descriptores tampoco ordenan las condiciones como el puntaje (el puntaje es máximo en 2 pasos; `d_hf` baja con los pasos).

**Cierre de H2.** Resultado neto: (i) el desplazamiento de sd-turbo sin cadena **no** se explica por el linaje SD del entrenamiento (§7); (ii) se reproduce con semillas y prompts nuevos pero es específico de la configuración y no monótono en los pasos (§9); (iii) no lo explican la nitidez ni la fracción de alta frecuencia (§11). **La causa sigue sin identificarse.** Candidatos vivos: propiedad concreta de sd-turbo con 2 pasos (C1 no descartada, pero sin dosis-respuesta), pipeline propio (C2) o rasgo no capturado por estos descriptores. Según §10, H2 no abre más etapas; una investigación posterior exigiría un generador adicional o más RAM, y sería otro estudio.
