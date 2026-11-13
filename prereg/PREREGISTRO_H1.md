# Preregistro H1 (borrador): huella física frente a IA, dos formulaciones

Estado: **borrador sin congelar**. Se congela (SHA-256 + etiqueta `prereg-h1-v1`) antes de ejecutar el análisis. La extracción (`bin/h1-extract.py`) solo guarda datos; no evalúa ni decide. Todo lo posterior a ver datos es exploratorio y se etiqueta así. Detector v1, registro de 81 K, `v3/` y código existente: solo lectura.

## 1. Hipótesis

- **H1-PRNU.** Una foto nativa contiene la K de su cámara; una imagen IA no contiene ninguna K del registro. Pregunta práctica: ¿sirve la presencia de K como señal de «real» y qué pasa con cámaras que no están en el registro?
- **H1-L2.** Una huella propia de **bajas frecuencias** (sin cámara conocida) separa reales nativos de imágenes IA de familias no vistas, tras igualar el procesado y superar los controles (L2: PREREGISTRO_L2_MALLET.md, H1-L2).

## 2. Datos (una sola extracción)

- Muestra de la sonda v1 `run3-multidevice`: 205 reales nativos de 41 dispositivos (JPEG, lado menor ≥ 2000 px, `nat`, ≤ 5 por dispositivo) + 600 IA Synthbuster (6 familias × 100). Rutas de `sample.json`, sin remuestrear.
- Reales Dresden (`dresden_*/nat`, ≤ 10 por cámara, semilla 20261111) como grupo **secundario** `real-dresden`: `trainingFiles` del registro está vacío, así que no se puede descartar solapamiento con las 15 fotos de la K. Solo en análisis de sensibilidad.
- Por imagen: NCC contra las 81 K (método de `v3/bin/registry_core.py`: recorte central nativo 512, píxeles almacenados sin EXIF-transpose, mediana 3×3 reflect, Pearson de gris); vector L1-proxy de 32 bins con y sin pipeline igualado (256 + JPEG Q90); log-potencia radial del residuo (64 bins log en [0,004; 0,5]); bytes, tamaño y formato.

## 3. Análisis fijado antes de ver los datos

**3.1 Mapeo dispositivo→K.** Por token de identificador (`modern_C05`, `vision_D02…`, `dresden_…`) contra los `entryId`. Se imprime la tabla y se revisa **sin mirar ninguna puntuación**. Los dispositivos sin K (p. ej. V2111) son «cámara ausente».

**3.2 H1-PRNU.**
- P0, control positivo de validez: AUC (propia-alta) de la K propia frente a K ajenas en reales con K. Si < 0,80, la extracción o el mapeo es inválido y se detiene el análisis.
- P1: puntuación `s = max NCC` sobre las 81 K. AUC real-alto frente a IA (a) con reales de cámara en el registro y (b) con reales de cámara **ausente**. TPR/FPR con el umbral 0,015 congelado en v3 (sin recalibrar).
- Lectura prevista: (a) alto y (b) ≈ 0,5 indicaría que PRNU solo identifica cámaras conocidas y no es un detector ciego. Se declara así si se cumple; no se rebusca otro umbral.

**3.3 H1-L2.**
- Rasgos: `lf_eq` (principal, procesado igualado) y `res_spec` (secundario, sin igualar).
- Modelo: regresión logística estandarizada. Validación: dejar una familia IA fuera (LOFO) y validación por grupos de dispositivo en los reales (ningún dispositivo en train y test a la vez). AUC macro por familia con IC95 bootstrap.
- Controles: (i) línea base de bytes/píxel + log(píxeles) + formato con la misma validación; (ii) permutación de etiquetas; (iii) identificación de dispositivo con los rasgos entre reales (si supera el azar con claridad, la señal es de dispositivo); (iv) sensibilidad quitando firefly (JPEG).
- Criterio (versión final, ver desviación H1-D1): **no apoyada** si AUC macro < 0,60 o límite inferior del IC95 ≤ 0,50. **Apoyada** si AUC macro ≥ 0,70, límite inferior > 0,60, AUC de firefly ≥ 0,60 (control de formato) y exactitud de identificación de dispositivo < 0,10 (azar ≈ 0,024). En otro caso, indeterminada. La línea base de bytes se informa, no decide.
- Validación: 5 pliegues de dispositivos × una familia IA fuera; la puntuación de cada IA es la media de sus 5 modelos; IC95 por bootstrap con reales remuestreados por dispositivo.

## 4. Limitaciones declaradas

- Reales: JPEG de móvil; IA: PNG (salvo firefly). La compresión previa es un confusor; el pipeline igualado lo mitiga, no lo elimina.
- 5 fotos por dispositivo: potencia baja a nivel de dispositivo; la estimación bootstrap trata las imágenes como unidades, con dependencia dentro del dispositivo.
- Los reales de cámara ausente dependen de qué cámaras tengan K en el registro (se conoce tras el mapeo).
- PRNU en IA con resolución nativa distinta de 512 usa el recorte central de 512 sin reescalar.
- No se evalúa nada de v1.

## 5. Registro de desviaciones

- **H1-D1** (antes de ver ningún dato extraído): el criterio «≥ 0,10 sobre la línea base de bytes» era inalcanzable por diseño, porque bytes/píxel y formato separan casi por completo reales JPEG de IA PNG. Se sustituye por el control de firefly (única familia IA en JPEG, comparada con reales JPEG): si el modelo aprendiera «JPEG», firefly saldría < 0,5. La línea base se mantiene como dato descriptivo. Se fijan además los umbrales numéricos de identificación de dispositivo (0,10) y de firefly (0,60).
- Pasos de ejecución: `h1-analyze.py --mapping-only` (tabla dispositivo→K sin puntuaciones) y después el análisis completo.

## 6. Resultados (run `h1-extract-run1`, `h1-analyze.py` en `53cce72`)

Plan, umbrales y H1-D1 fijados antes de ver puntuaciones. Mapeo revisado antes del análisis: 49 mapeados, 8 ausentes (V2111, C10, C11, C12, C16, C17, C18, D28), 2 ambiguos (D12, D21; excluidos por regla). Reales de `run3`: 155 con K, 40 de cámara ausente, 10 excluidos. 905 + 90 Dresden = 995 registros, 0 errores. Los JSON están solo en el servidor.

**6.1 H1-PRNU**
- P0 (control positivo): AUC 0,9425, top1 0,781 → VÁLIDO.
- P1(a) real con K (n=155): AUC 0,931 [0,905; 0,954]; fracción ≥ 0,015: 0,755; IA: 0,080.
- P1(b) cámara ausente (n=40): AUC 0,587 [0,494; 0,675]; fracción ≥ umbral 0,125.
- IA por familia (≥ umbral): firefly 0,56; resto 0,00.
- Sensibilidad Dresden (posible solape con K, n=90): AUC 0,823.
- Lectura: la PRNU identifica cámaras conocidas y no es detector ciego (opción (b) del plan). El 56 % de firefly es sospechoso de artefacto de formato o recompresión (no verificado).

**6.2 H1-L2** (5 pliegues de dispositivo × LOFO; 205 reales, 41 dispositivos, 700 IA)
- `lf_eq`: macro AUC 0,799 [0,776; 0,821]; permutación 0,500; firefly 0,657; sin firefly 0,827; bytes 0,993; identificación de dispositivo 0,059 (azar 0,024). Veredicto por regla: **apoyada**.
- `res_spec` (secundario): macro 0,901 [0,885; 0,915]; firefly 0,599 (< 0,60); sin firefly 0,961; identificación 0,078. Veredicto: **indeterminada**.
- Reservas: la línea base de bytes (0,993) separa casi todo; el único control de formato es firefly (0,657 en `lf_eq`, 0,599 en `res_spec`). El salto de `res_spec` al quitar firefly sugiere formato JPEG/PNG. `lf_eq` es más creíble porque iguala el procesado. Todo exploratorio, un solo corpus, 5 fotos por dispositivo.

**6.3 Pendiente (no ejecutado):** control de recompresión idéntica (mismo JPEG para reales e IA) repitiendo `res_spec`/`lf_eq`; verificar causa del 56 % en firefly.

## 7. Controles de formato y geometría (fijados antes de ejecutar; scripts `h1-controls-extract.py`, `h1-post.py`)

**Motivo.** En la sección 6 la línea base de bytes separa 0,993 y el único control de formato es firefly. Confusores no descartados: (i) historia de compresión (reales JPEG de cámara, IA PNG); (ii) rejilla JPEG 8×8: el recorte central de 512 de los reales puede no estar alineado a 8, el de las IA (lados múltiplos de 64) sí; (iii) escala: reales ≥ 2000 px, IA ≈ 768–1344 px.

**Brazos** (mismas 905 imágenes, mismas funciones de rasgos): `mod8` (recorte alineado a 8), `q75` y `q92` (mod8 + JPEG 4:2:0 a TODAS las imágenes), `down` (reales reducidos a lado menor 1024, IA sin tocar), `down_q92`. Misma validación (5 pliegues de dispositivo × LOFO), `n_boot` 300, sin permutación.

**Criterio, por rasgo (`lf_eq`, `res_spec`):**
- **Sobrevive**: en `mod8`, `q75` y `q92` macro AUC ≥ 0,70, límite inferior del IC95 > 0,60 y AUC de firefly ≥ 0,60.
- **Colapsa**: macro < 0,60 o límite inferior ≤ 0,50 en `q75` o en `q92`.
- **Parcial**: en otro caso.
- Se informa si la igualación fue efectiva (línea base de bytes < 0,80 tras recomprimir). `down*` es solo sensibilidad: reduce solo a los reales y puede crear un confusor inverso.

**Predicción (antes de ver datos).** `res_spec` colapsa (su salto al quitar firefly apunta a formato). `lf_eq` pierde parte de su señal y queda «parcial». Si `lf_eq` sobrevive en `q75`/`q92`, es el primer indicio con formato igualado.

**Análisis posteriores sobre `h1-extract-run1` (exploratorios, sin criterio previo):** alineación 8×8 y tamaño por grupo; K ganadora y tipo de K en firefly (hipótesis: el bloqueo JPEG de las K correlaciona con imágenes JPEG, no con PRNU); reales JPEG frente a firefly JPEG con validación por dispositivo; AUC por bin de frecuencia (si la señal está en armónicos de 1/8 ciclo/px, es rejilla JPEG).

## 8. Resultados de los controles (`h1-controls-run1` y `h1-post.py`, commit `0799ad1`)

905 imágenes, 0 errores. Veredicto del criterio de la sección 7: `lf_eq` **parcial**, `res_spec` **parcial**.

| brazo | `lf_eq` macro [IC95] / firefly | `res_spec` macro [IC95] / firefly | base bytes |
|---|---|---|---|
| mod8 | 0,784 [0,760; 0,805] / 0,640 | 0,900 [0,884; 0,914] / 0,599 | 0,993 |
| q75 | 0,877 [0,859; 0,893] / 0,694 | 0,916 [0,899; 0,928] / 0,680 | 0,733 |
| q92 | 0,685 [0,657; 0,713] / 0,516 | 0,898 [0,878; 0,911] / 0,689 | 0,732 |
| down (sensibilidad) | 0,581 [0,558; 0,607] / 0,389 | 0,774 [0,753; 0,794] / 0,505 | 0,993 |
| down_q92 (sensibilidad) | 0,581 [0,558; 0,604] / 0,353 | 0,815 [0,795; 0,832] / 0,721 | 0,646 |

- **Predicción de la sección 7:** `lf_eq` parcial se cumple. `res_spec` colapsa **queda refutada**: aguanta la recompresión (macro ≈ 0,90, firefly ≈ 0,68). Es «parcial» solo porque firefly en `mod8` da 0,599, a 0,001 del umbral.
- **Geometría:** recorte alineado a 8 en el 85 % de los reales y el 100 % de las IA. Alinear apenas mueve los resultados (`lf_eq` 0,799 → 0,784), así que la rejilla 8×8 no explica la señal.
- **Escala:** lado menor mediano ≈ 3000 en reales frente a 512–1792 en IA. Al reducir los reales a 1024, `lf_eq` cae a 0,58 (firefly < 0,5) y `res_spec` a 0,77–0,82.
- **`lf_eq` en q92:** 0,685, por debajo de la línea base de bytes (0,732).
- **Firefly y PRNU:** el 56 % supera el umbral de PRNU, y solo 8 K ganan en las 100 imágenes (Pentax_OptioA40: 53). Es un patrón genérico de imagen JPEG, no identificación de cámara.
- **Mismo formato (JPEG real frente a firefly):** `lf_eq` 0,758, `res_spec` 0,924, bytes/píxel 1,000. Separa, pero la línea base separa perfecto, así que no prueba huella física.
- **Bins:** la separación de `res_spec` frente a IA PNG se concentra en 0,36–0,48 ciclos/px (AUC 0,84–0,87, cerca de Nyquist). En firefly está en 0,004–0,01. Esto es consistente con un efecto de escala o de procesado, y no lo identifica.

### Conclusión de H1 (exploratoria, un corpus, 5 fotos por dispositivo)
1. **H1-PRNU:** identifica cámaras registradas (P0 0,94; AUC real con K 0,931). **No** es detector ciego (cámara ausente 0,587 [0,494; 0,675]). El resultado en firefly es un artefacto.
2. **H1-L2, `lf_eq` (baja frecuencia):** **no robusta.** Sobrevive a alinear y a Q75, se degrada en Q92 y desaparece al igualar escala.
3. **`res_spec` (alta frecuencia del residuo):** persiste tras recompresión y parcialmente tras igualar escala. La causa no está identificada: puede ser huella de generación o diferencia de procesado y escala entre móvil e IA. No se puede afirmar nada con este diseño.
4. **Límite del diseño:** reales y IA difieren a la vez en contenido, formato, escala y procesado. Ningún control con estos datos los separa del todo.

**Siguiente prueba propuesta (no ejecutada):** reales reconstruidos con el autocodificador (VAE) de un generador (mismo contenido, mismo formato y escala, pasan por el decodificador de la IA) frente a los mismos reales. Aísla la huella del decodificador de todos los confusores anteriores.

## 9. Prueba del decodificador (fijada antes de ejecutar; `h1-vae-recon.py`)

**Pregunta.** ¿El decodificador de un generador deja en `res_spec`/`lf_eq` una huella distinta de la de un simple paso bajo o una recompresión, con el contenido, el formato y la escala del real intactos?

**Diseño.** Los 205 reales de `run3`: recorte 512 alineado a 8 en píxeles nativos, sin reescalar. Cuatro brazos con el mismo contenido: `crop`, `recon` (VAE `stabilityai/sd-vae-ft-mse`, media del latente), `blur` (gaussiano σ = 0,7, control inespecífico) y `jpeg75` (control de recompresión). Rasgos idénticos a los de la sección 7.

**Medidas.** (A) AUC por dispositivo (5 pliegues) de `recon` frente a `crop`, IC95 por bootstrap de dispositivos. (B) Modelo real-frente-a-IA entrenado con `h1-controls-run1` (brazo `mod8`, solo reales de otros pliegues): fracción de pares en que `recon` puntúa más «IA» que su `crop` (Wilcoxon pareado), comparada con `blur` y `jpeg75`.

**Criterio (`res_spec`).** *Apoyada* si el límite inferior del IC95 de (A) > 0,90, (B) ≥ 80 % de pares con p < 0,001, y el desplazamiento mediano de `recon` supera el de `blur` y el de `jpeg75`. *No apoyada* si falla (A) o (B). *Indeterminada* si pasa A y B pero no supera a los controles. `lf_eq` se informa sin criterio.

**Predicción (antes de ver datos).** Un VAE suaviza las altas frecuencias del residuo, así que (A) separará casi seguro. La duda real es si `recon` se acerca a las IA más que el desenfoque. Espero «indeterminada».

**Limitaciones declaradas.** Solo un VAE (el de SD 1.x/2.x); no representa a DALL·E, Midjourney ni Firefly. El VAE trabaja a 512 px sobre píxeles nativos de móvil, más detallados que las imágenes de entrenamiento. Aunque salga «apoyada», solo muestra que el decodificador es capaz de dejar esa huella, no que sea la causa de la separación en el corpus.

## 10. Reales reconstruidos con VAE (EXPLORATORIO; sin criterio fijado antes; `h1-vae-auc.py`, commit `598ac42`)

**Qué se hizo.** El usuario pasó 50 imágenes reales de `dataset_test/real/` (512×512 PNG, procedencia COCO según el usuario, no verificada aquí) por un VAE (script propio del usuario, no versionado) y se guardaron como `real_vae_000…049`. Emparejado por orden alfabético, verificado por miniatura (50/50, correlación mínima 0,998). No es la prueba de la sección 9 (`h1-vae-recon.py`, no ejecutada, criterio propio). IA: 700 imágenes de `h1-controls-run1` (brazo `mod8`). AUC de B y C = «IA > real». IC95 por bootstrap de imágenes (no hay dispositivos: dependencia no modelada).

| Escenario | `res_spec` | `lf_eq` |
|---|---|---|
| A) VAE vs original | 0,586 [0,536; 0,645] | 0,565 [0,506; 0,631] |
| B) IA vs VAE, entrenado con los 205 reales de `run3` | 0,613 [0,547; 0,685] | 0,677 [0,594; 0,753] |
| C) IA vs original, ídem | 0,624 [0,554; 0,692] | 0,618 [0,540; 0,690] |
| B) IA vs VAE, en dominio (5 pliegues) | 0,841 [0,791; 0,888] | 0,662 [0,583; 0,734] |
| C) IA vs original, en dominio | 0,873 [0,826; 0,913] | 0,650 [0,574; 0,727] |

**Lectura.**
- Si el decodificador del VAE dejara la huella que separa a la IA, los reales reconstruidos puntuarían como IA (B cerca de 0,5 o inferior). No ocurre: B ≈ C (diferencia 0,03 en `res_spec` en dominio). El VAE mueve los reales muy poco hacia la IA (A = 0,586).
- Por tanto, la separación real/IA de `res_spec` **no se explica por el paso por un VAE** (este VAE, a 512 px). No se sigue de ahí que sea una «huella de generación»: tampoco descarta diferencias de procesado, escala, compresión previa o contenido entre corpus.
- En dominio `res_spec` separa 0,87, pero con reales de `run3` baja a 0,62. La explicación «cámaras distintas» es una hipótesis no verificada. La alternativa es que el modelo en dominio aprende a distinguir **el corpus** de los reales (reducción previa, JPEG de origen) de las IA.

**Estado de H1-L2 (sin cambio):** `lf_eq` parcial, `res_spec` parcial (sección 8). Este resultado no la eleva a «apoyada»; solo descarta una explicación (el VAE) de las posibles.

**Límites.** Un solo VAE; 50 imágenes; procedencia de los reales sin verificar; script de reconstrucción no versionado; resultado post hoc.

## 11. Prueba LOCO (leave-one-corpus-out) — criterios y predicción fijados ANTES de ejecutar

**Motivo.** La sección 10 muestra que `res_spec` rinde 0,87 en dominio y 0,62 fuera de dominio. Se pregunta si separa «real/IA» o solo «qué corpus de reales es». Procedencia declarada por el autor: `dataset_test/real` = 50 imágenes COCO reescaladas a 512×512 (probablemente PIL Lanczos), PNG, script no conservado; `run3` = 205 fotos nativas de móvil (JPEG); Dresden = JPEG de cámara.

**Diseño** (`bin/h1-loco.py`, autotest incluido). Corpus de reales: `coco` (50), `run3` (205, 41 dispositivos), `dresden` (≤10/cámara, regla y semilla de H1). Positiva: 700 IA Synthbuster (7 familias; modelo LOFO por familia como en secciones 6 y 8). Pipeline común P: recorte cuadrado central → LANCZOS a 512 (si no es 512) → mismos rasgos.
- Brazo **igual (decisivo)**: reales e IA pasan por P.
- Brazo **nativo (referencia)**: reales por P; IA con recorte 512 nativo (`h1-controls-run1`, mod8).
- Pruebas: (1) entrenar con un corpus y evaluar en cada otro (6 pares); (2) entrenar con dos y evaluar en el tercero; (3) referencia en dominio (5 pliegues por dispositivo; COCO = 1 imagen por clúster); (4) identificación del corpus solo con reales (exactitud balanceada, azar 0,33). IC95 por bootstrap sobre dispositivos (reales) e imágenes (IA).

**Criterio (brazo igual, por rasgo, sobre los 6 pares):** «aguanta» si media ≥ 0,75 y el menor límite inferior del IC95 ≥ 0,60; «artefacto de corpus» si media < 0,68; en otro caso «indeterminada». Apoyo: identificación del corpus ≥ 0,80 indica que el rasgo codifica el corpus.

**Predicción (fijada antes de ver datos).** `res_spec`: «artefacto de corpus» o «indeterminada» (p ≈ 0,65); `lf_eq`: «artefacto de corpus» (p ≈ 0,75). Incertidumbre: puede haber señal real además del corpus.

**Lectura permitida.** Un «aguanta» sería solo un candidato: sigue habiendo un solo generador de IA (Synthbuster), 5 fotos por dispositivo y análisis exploratorio; exigiría confirmación preregistrada en un corpus nuevo antes de escribir nada. Un «artefacto de corpus» cierra H1-L2 como no apoyada para estos rasgos. Los resultados cruzados solo son comparables si COCO se reescaló de forma parecida; esa duda queda como limitación.

## 12. Resultados LOCO (`h1-loco.py`, commit `ab6e3f3`, 1045 imágenes, 0 errores)

Reales: coco 50, run3 205, dresden 90; IA 700. IC95 entre corchetes.

| Brazo / rasgo | Media cruzado (6 pares) | Mín. IC inf. | Rango de pares | Veredicto |
|---|---|---|---|---|
| igual / `res_spec` | 0,536 | 0,353 | 0,421–0,616 | artefacto de corpus |
| igual / `lf_eq` | 0,502 | 0,384 | 0,441–0,568 | artefacto de corpus |
| nativo / `res_spec` | 0,714 | 0,524 | 0,580–0,844 | indeterminada |
| nativo / `lf_eq` | 0,524 | 0,425 | 0,486–0,574 | artefacto de corpus |

- LOCO (dos corpus → el tercero), igual `res_spec`: 0,595 / 0,603 / 0,588; `lf_eq`: 0,545 / 0,494 / 0,432.
- En dominio (igual) `res_spec`: coco 0,897, run3 0,593, dresden 0,723.
- Identificación del corpus solo con reales: `res_spec` 0,526, `lf_eq` 0,453 (azar 0,333; umbral 0,80). No se alcanza el umbral.

**Predicción:** cumplida para ambos rasgos (`res_spec` artefacto/indeterminada; `lf_eq` artefacto).

**Conclusión (fijada por criterio).** Con el mismo procesado en reales e IA, ningún rasgo generaliza entre corpus de reales (AUC ≈ 0,5–0,6). H1-L2 queda **no apoyada** para `lf_eq` y `res_spec`. La separación vista antes (0,8–0,9) depende del procesado/corpus: en el brazo nativo `res_spec` sube a 0,58–0,84, justo cuando la IA no se reescala y los reales sí.

**Matices.**
- El rasgo no codifica el corpus (identificación ≈ azar), así que «artefacto de corpus» significa «sin transferencia», no «identifica el corpus».
- La alta cifra en dominio de COCO (0,897) con transferencia nula indica ajuste al propio corpus de 50 imágenes.
- Limitación: P reduce `run3` (~3000 px) a 512 con Lanczos y puede borrar una señal que exista a resolución nativa. Esta prueba no la descarta; solo que no sobrevive a procesado igualado.
- Los hallazgos de la sección 10 (VAE) no se pueden leer como huella de generación.

## 13. Prueba LOCO a resolución NATIVA — criterios y predicción fijados ANTES de ejecutar

**Motivo.** El pipeline P de la sección 11 reduce `run3` (~3000 px) a 512 y puede borrar una señal nativa. Esta prueba no reescala.

**Diseño** (`bin/h1-native.py`, autotest incluido). Solo corpus nativos: `run3` (205, de `h1-controls-run1`) y `dresden` (90, extraídos aquí con la regla y semilla de H1). COCO queda fuera (ya viene a 512). IA: las 700 de `h1-controls-run1`. Recorte 512 alineado a 8 **sin reescalar**. Brazos: `mod8` (referencia), `q75` y `q92` (JPEG 4:2:0 a TODAS las imágenes; iguala formato). Pares cruzados: `run3→dresden` y `dresden→run3`; LOFO por familia; IC95 por dispositivo/imagen.

**Criterio (brazo decisivo q92, por rasgo, sobre los 2 pares):** mismos umbrales que la sección 11: «aguanta» si media ≥ 0,75 y menor IC inf. ≥ 0,60; «artefacto de corpus» si media < 0,68; si no, «indeterminada». `q75` y `mod8` son sensibilidad.

**Predicción (antes de ver datos).** `res_spec`: indeterminada o artefacto (p ≈ 0,7); `lf_eq`: artefacto (p ≈ 0,8). Solo 2 pares → poca potencia.

**Lectura permitida.** «Aguanta» solo sería candidato y exigiría confirmación en un corpus nuevo preregistrado. Un resultado negativo cierra H1-L2 también a resolución nativa. Limitaciones: solo 2 corpus; ambos de cámaras con JPEG propio; IA solo Synthbuster.

## 14. Resultados LOCO nativo (`h1-native.py`, commit `e78e3ca`; 205 run3 + 90 dresden, 700 IA)

| Brazo / rasgo | run3→dresden | dresden→run3 | Media | Mín. IC inf. | Veredicto |
|---|---|---|---|---|---|
| q92 `res_spec` (decisivo) | 0,932 [0,903, 0,957] | 0,863 [0,833, 0,891] | 0,897 | 0,833 | **aguanta** |
| q92 `lf_eq` (decisivo) | 0,779 [0,731, 0,831] | 0,687 [0,620, 0,751] | 0,733 | 0,620 | indeterminada |
| q75 `res_spec` / `lf_eq` | 0,917 / 0,890 | 0,886 / 0,861 | 0,901 / 0,876 | 0,853 / 0,828 | aguanta / aguanta |
| mod8 `res_spec` / `lf_eq` | 0,910 / 0,840 | 0,869 / 0,774 | 0,890 / 0,807 | 0,838 / 0,717 | aguanta / aguanta |

**Predicción refutada** (`res_spec` indeterminada/artefacto p≈0,7; `lf_eq` artefacto p≈0,8). Resultado exploratorio hasta confirmación.

**Lectura.** A resolución nativa y con formato igualado, `res_spec` transfiere entre dos corpus de cámaras (≈0,86–0,93); con procesado igualado a 512 (sección 12) se hunde a ≈0,54. La señal depende del pixel-scale del contenido, no del formato.

**Confusión no resuelta: escala de contenido.** Un recorte de 512 px de una foto de ~3000 px es una zona ampliada, suave y de poco detalle por píxel; una IA a 512 es la escena entera. Ambos corpus «aguantan» entre sí porque comparten esa propiedad (foto grande recortada), no porque se haya aislado la generación. Consistente con la sección 8 (reducir los reales a 1024: `res_spec` 0,77–0,82, `lf_eq` 0,58) y con la 12 (reducir a 512: ≈0,54). `firefly` (JPEG, 1792 px, la IA más parecida en formato y escala) es la familia más baja en secciones 6 y 8 (`res_spec` 0,60–0,72).

**Estado de H1-L2:** no apoyada como huella de generación; `res_spec` queda como **señal dependiente de escala/captura pendiente de control de dosis-respuesta**. No se escribe paper.

## 15. Curva dosis-respuesta de escala — criterios y predicción fijados ANTES de ejecutar

**Motivo.** La sección 14 deja abierta una confusión: a escala nativa `res_spec` aguanta entre `run3` y Dresden (media 0,897), pero a 512 con procesado igualado cae a ≈ 0,54 (sección 12). Se pregunta si la señal es función de la escala del contenido (cuánto se reduce el real antes del recorte) o persiste.

**Diseño** (`bin/h1-scale.py`, autotest de 5 comprobaciones incluido). Reales `run3` (205) y Dresden (90, regla y semilla de H1) reducidos con LANCZOS por f ∈ {1; 1,5; 2; 2,5; 3} antes del recorte 512 alineado a 8; extremo «P»: reducidos a lado menor 512 (como la sección 11). JPEG q92 4:2:0 a todas las imágenes de los reales; IA = brazo q92 de `h1-controls-run1`, sin tocar. **Muestra común**: solo reales con lado menor ≥ 512 × 3 = 1536 px, la misma en todos los factores; se informan los excluidos. Medida: AUC cruzado `run3→dresden` y `dresden→run3` (LOFO por familia, IC95 como en la sección 11) en cada factor; curva = media de los dos pares, ordenada de f = 1 a P.

**Criterio (por rasgo):**
- «dependiente de escala»: AUC(f=1) − AUC(P) ≥ 0,20 y la curva no sube más de 0,03 entre puntos consecutivos;
- «no explicada por escala»: AUC(P) ≥ 0,75;
- «indeterminada» en otro caso.

**Predicción (antes de ver datos).** `res_spec`: «dependiente de escala» (p ≈ 0,7). `lf_eq`: «dependiente de escala» o «indeterminada».

**Lectura permitida.** «Dependiente de escala» cierra H1-L2 como no apoyada: la separación observada es una propiedad de la captura (foto grande recortada frente a imagen entera), no una huella de generación. «No explicada por escala» no probaría generación: exigiría un control de remuestreo (los reales reducidos llevan el filtro de paso bajo de LANCZOS, las IA no) y confirmación en corpus nuevo. Limitaciones: reales solo de dos corpus; IA solo Synthbuster; la reducción deja huella de remuestreo que la IA no tiene.

## 16. Resultados de la curva de escala (`h1-scale.py`, commit `fea4c86`; 205 run3 + 90 dresden, 0 excluidos, 0 errores)

Media de los dos pares cruzados (`run3↔dresden`), JPEG q92 a todas; IC95 en el log del servidor.

| Brazo | x1 | x1,5 | x2 | x2,5 | x3 | P (a 512) | Veredicto |
|---|---|---|---|---|---|---|---|
| `res_spec` | 0,897 | 0,940 | 0,898 | 0,842 | 0,823 | 0,781 | **no explicada por escala** (P ≥ 0,75; caída 0,12 < 0,20) |
| `lf_eq` | 0,733 | 0,733 | 0,614 | 0,536 | 0,520 | 0,509 | **dependiente de escala** (caída 0,22, monótona) |

**Predicción:** `res_spec` «dependiente de escala» (p≈0,7): **refutada**. `lf_eq`: cumplida.

**Lectura.** `lf_eq` es una propiedad de escala: se cierra como no apoyada. `res_spec` decae suavemente (0,94 → 0,78) pero no se hunde, y en P conserva 0,83/0,73 entre corpus.

**Discrepancia sin resolver.** La sección 12 (también reducción a 512, sin JPEG añadido) dio 0,54 para `res_spec`; aquí, con JPEG q92 en todas, 0,78. Difieren el JPEG añadido, el recorte alineado a 8 y la IA (q92 frente a sin tocar). Hay que aclarar cuál de estos factores devuelve la señal antes de interpretarla.

**Lo que NO se concluye.** No es evidencia de huella de generación. Confusiones abiertas: (i) los reales reducidos llevan el paso bajo de Lanczos y las IA no; (ii) los reales tienen doble historial JPEG y las IA PNG→JPEG una sola vez. Exploratorio hasta pasar un control de remuestreo y de doble compresión y confirmar con corpus nuevo.

## 17. Control de remuestreo y doble compresión — criterios y predicción fijados ANTES de ejecutar

**Motivo.** La sección 16 deja dos confusiones: (i) los reales reducidos llevan el paso bajo de Lanczos y las IA no; (ii) los reales tienen doble historial JPEG (cámara + q92) y las IA PNG→q92. Además `res_spec` a 512 da 0,54 (§12, sin JPEG añadido) y 0,78 (§16, con q92 en todas).

**Diseño** (`bin/h1-resample.py`, autotest de 4 comprobaciones). Mismos reales (`run3` 205 + Dresden 90) reducidos a lado menor 512 (Lanczos), recorte alineado a 8: R0 sin JPEG, R1 con JPEG q92. IA (700): recorte 512 alineado a 8 y cadenas que imitan al real: A0 nada; A1 q92; A2 JPEG q90 + q92 (doble compresión); A3 Lanczos ×1,5 arriba y abajo (paso bajo); A4 = A3 + q92; A5 = A3 + q90 + q92 (todo igualado); A6 = A3 + q90 (todo igualado sin JPEG final). Comparaciones (IA/real): C1 A0/R0, C2 A1/R1, C3 A2/R1, C4 A4/R1, **C5 A5/R1 (decisiva)**, C6 A6/R0. AUC cruzado `run3↔dresden`, LOFO por familia, IC95 como en la sección 11.

**Criterio (C5, por rasgo):** «persiste con procesado igualado» si media ≥ 0,75 y menor IC inf. ≥ 0,60; «explicada por procesado» si media < 0,68; si no, «indeterminada». C1–C4 y C6 atribuyen el efecto (qué factor lo devuelve) y son descriptivos.

**Predicción (antes de ver datos).** `res_spec`: «explicada por procesado» (p ≈ 0,6); `lf_eq`: «explicada por procesado» (p ≈ 0,85).

**Autotest.** Con IA hecha por la misma cadena que los reales (incluido el JPEG nativo) C1 ≈ 0,54; con huella inyectada C5 = 1,0. Hallazgo del autotest: en contenido de ruido, el JPEG nativo previo del real por sí solo separaba a 1,0 de una IA sin ese historial, es decir, el historial JPEG es potente como confusor.

**Lectura permitida.** «Persiste» sería candidato a confirmar en corpus nuevo con otro generador, no prueba de huella de generación (las cadenas imitan, no reproducen, el historial real: el paso bajo de Lanczos ×1,5 no equivale al de una reducción ~6×). «Explicada» cierra `res_spec` como no apoyada. Limitaciones: Synthbuster únicamente; dos corpus de reales; el historial de IA como Midjourney/DALL·E (reescalados internos) es desconocido.

## 18. Resultados del control de remuestreo (`h1-resample.py`, commit `48ec33b`; 700 IA, 205 run3 + 90 dresden, 0 errores)

Media cruzada `run3↔dresden` (IA/real):

| Comp. | `res_spec` | `lf_eq` |
|---|---|---|
| C1 A0/R0 (nada / sin JPEG) | 0,768 | 0,488 |
| C2 A1/R1 (q92 / q92) | 0,781 | 0,509 |
| C3 A2/R1 (doble JPEG) | 0,730 | 0,464 |
| C4 A4/R1 (paso bajo + q92) | 0,811 | 0,540 |
| **C5 A5/R1 (todo igualado, decisiva)** | **0,856** (mín. IC inf. 0,797) | **0,631** (0,542) |
| C6 A6/R0 | 0,822 | 0,564 |

**Veredictos:** `res_spec` «persiste con procesado igualado»; `lf_eq` «explicada por procesado».
**Predicciones:** `res_spec` explicada (p≈0,6): **refutada**; `lf_eq` explicada (p≈0,85): cumplida.

**Lectura.** `lf_eq` queda cerrada como no apoyada (≈0,5–0,63). En `res_spec`, imitar el procesado del real (paso bajo, doble JPEG) no reduce la separación, la aumenta (0,77 → 0,86). Los dos confusores específicos preregistrados (paso bajo de Lanczos, doble JPEG) no la explican.

**Discrepancia con §12 (0,54): hipótesis, no resultado.** En §12 la IA también pasaba por Lanczos a 512 (firefly 1792 px, SDXL 896 px…), aquí la IA va en píxeles nativos. Es coherente con que la señal viva en los píxeles nativos de la IA y desaparezca al reescalarla, pero no se ha probado.

**Lo que NO se concluye.** No es una huella de generación demostrada. Sigue sin controlar: un único benchmark y generador de IA conocido (Synthbuster), reales siempre reducidos ≥ 3× y la IA no (asimetría de escala estructural: no existe real nativo de 512 en estos corpus salvo COCO, ya reescalado), el historial de las IA (reescalados internos) desconocido, y el paso bajo de ×1,5 no equivale al de una reducción ~6×. Estado: **candidato exploratorio para confirmación preregistrada** en corpus nuevo y con otro generador (propio, con historial conocido).

## 19. Prueba con IA propia de historial conocido (criterios y predicción fijados ANTES de ejecutar)

**Objetivo.** Confirmar o descartar que `res_spec` sea una propiedad de Synthbuster y no de la generación, con imágenes cuyo historial se conoce (sin reescalado ni JPEG).

**Generación** (`bin/h1-owngen-make.py`, autotest 4/4): 512×512 PNG nativo. Primario `sdturbo` (stabilityai/sd-turbo, 2 pasos, guidance 0, n=100). Secundario `sd15` (n=40, 20 pasos, guidance 7,5, safety_checker desactivado), misma familia que sd1 de Synthbuster, por tanto NO independiente. Prompts y semillas fijos en el script (SEED0=20261112); manifiesto con SHA-256 y versiones. Si una imagen no es 512×512 se aborta.

**Evaluación** (`bin/h1-owngen-eval.py`, autotest 3/3). Entrenamiento: 700 IA Synthbuster + reales del corpus X; test: IA propia + reales del corpus Y (run3→dresden y dresden→run3). Brazo **primario A5/R1** (IA con paso bajo ×1,5 + q90 + q92 frente a reales reducidos + q92), sensibilidad A0/R0 y A1/R1 (cada uno con su brazo Synthbuster). AUC IA propia > reales de test; IC95 bootstrap (imágenes IA, dispositivos reales).

**Criterios.** Mismos umbrales: «se confirma» si media ≥ 0,75 y mín. IC inf. ≥ 0,60; «propiedad de Synthbuster» si media < 0,68; resto indeterminada. Veredicto principal: `res_spec`, sdturbo, brazo primario. `lf_eq` solo informativo (ya cerrada).

**Predicción.** `res_spec` se confirma con p≈0,4.

**Limitaciones.** CPU, n pequeño, un generador independiente, sin control de bytes, reales siempre reducidos (asimetría de escala persistente), safety_checker desactivado. Un resultado positivo es solo candidato; exigiría corpus real nuevo. Un resultado negativo cierra `res_spec` como propiedad de Synthbuster.

**Resultados:** pendientes (se registrarán en §20 sin modificar esta sección).

**Desviación H1-D2 (registrada antes de generar nada).** El servidor tiene 3,8 GB de RAM; sd-turbo en float32 (~5 GB) entró en swap continuo y se abortó sin producir imágenes (0 ficheros). Se genera en bfloat16 con `low_cpu_mem_usage` (~2,6 GB). Efecto: los pesos y cálculos tienen menor precisión que en float32, y las imágenes difieren numéricamente de una generación float32 con la misma semilla; el dtype queda en el manifiesto. No cambia criterios ni predicción. `sd15` (secundario, ~4 GB en float32) queda fuera salvo que haya memoria suficiente.

**Desviación H1-D3 (registrada antes de generar nada).** Con bfloat16 (H1-D2), el servidor estuvo >10 min de CPU al 100 % sin producir ninguna imagen (0 ficheros); se abortó. Con pm2 detenido para liberar RAM, se genera en float32 con `enable_attention_slicing` y `enable_vae_slicing` (el slicing no altera el resultado salvo error numérico mínimo). El dtype queda en el manifiesto. Sustituye a H1-D2; criterios y predicción sin cambios.

Corrección técnica de H1-D3: `pipe.enable_vae_slicing()` no existe en la versión de diffusers del servidor; se usa `pipe.vae.enable_slicing()` (mismo efecto). El primer intento abortó con AttributeError sin generar imágenes.

## 20. Resultados de la prueba con IA propia (§19; sdturbo n=100, float32+slicing, H1-D3; `results/h1-owngen-run1`)

Commit de código esperado: `9d95624` (el pegado del servidor no incluyó `git log`).

| rasgo | brazo | run3→dresden | dresden→run3 | media | veredicto |
|---|---|---|---|---|---|
| res_spec | **primario A5/R1** | 0,946 [0,908; 0,975] | 0,939 [0,907; 0,966] | **0,942** | **se confirma** |
| res_spec | A0/R0 | 0,753 [0,685; 0,818] | 0,655 [0,586; 0,723] | 0,704 | indeterminada |
| res_spec | A1/R1 | 0,516 [0,427; 0,611] | 0,503 [0,438; 0,570] | 0,509 | propiedad de Synthbuster |
| lf_eq | primario A5/R1 | 0,888 [0,810; 0,948] | 0,882 [0,845; 0,915] | 0,885 | se confirma |
| lf_eq | A0/R0 | 0,210 [0,137; 0,291] | 0,292 [0,236; 0,350] | 0,251 | propiedad de Synthbuster |
| lf_eq | A1/R1 | 0,185 [0,120; 0,262] | 0,252 [0,199; 0,308] | 0,219 | propiedad de Synthbuster |

**Veredicto preregistrado:** H1-L2-IAPROPIA `res_spec` (primario, sdturbo): **se confirma**. Predicción (p≈0,4): refutada.

**Lectura exploratoria (post hoc, no preregistrada).** El resultado depende de la cadena de procesado aplicada a la IA: con A1/R1 la IA propia está en azar (0,509) aunque Synthbuster daba 0,781 en §18; solo con A5 (paso bajo + q90 + q92, que los reales no reciben) sube a 0,942. `lf_eq` cambia de 0,22–0,25 (invertido) a 0,885 solo por la cadena, y ya estaba explicado por procesado (§18); su «se confirma» no se interpreta como hallazgo. Hipótesis sin probar: el modelo entrenado en A5 detecta la cadena A5 y no la generación. Probabilidad subjetiva: ~35 % de señal genuina residual.

**No se concluye** huella de generación. Limitaciones: un generador, n=100, float32 con slicing, reales siempre reducidos, `sd15` no generado, cadena A5 impuesta solo a la IA.

**Siguiente (a preregistrar en §21 antes de ejecutar):** (a) modelo entrenado en A5, IA propia A5 vs IA propia A1 (si AUC≈1, detecta la cadena); (b) reales con A5 (R5) vs R1.

## 21. Control de cadena (criterios y predicciones fijados ANTES de ejecutar)

**Motivo.** En §20, `res_spec` se separa con IA propia solo en el brazo A5/R1 (0,942) y no en A1/R1 (0,509). La IA recibe la cadena A5 (paso bajo ×1,5 + q90 + q92) y los reales no. Hipótesis a contrastar: el modelo detecta la cadena y no la generación.

**Diseño** (`bin/h1-chain.py`, autotest 6/6; no genera imágenes). Se aplica a los reales la misma cadena (R5 = real reducido a 512, paso bajo ×1,5, q90, q92). Se reutilizan los rasgos de Synthbuster (A5), reales (R1) e IA propia (A1, A5). Pares run3↔dresden; IC95 por bootstrap (dispositivos en reales, imágenes en IA).

- **(c) PRIMARIO, cadena igualada en ambos lados:** entrena Synthbuster A5 + R5(X); test IA propia A5 vs R5(Y). Veredicto con los umbrales de siempre: «persiste con cadena igualada» (media ≥ 0,75 y mín. IC inf. ≥ 0,60), «artefacto de cadena» (media < 0,68), resto indeterminada.
- **(a) diagnóstico:** modelo A5/R1; AUC IA propia A5 > IA propia A1 (mismas imágenes). ≥ 0,75 con mín. IC inf. ≥ 0,60 = «cadena detectada»; < 0,68 = «cadena no detectada».
- **(b) diagnóstico:** modelo A5/R1; AUC R5 > R1 (mismos reales). Mismos umbrales.

**Lectura preregistrada.**
- (c) artefacto de cadena (con (a) o (b) «cadena detectada»): `res_spec` queda cerrado como artefacto de cadena; no hay evidencia de huella de generación.
- (c) persiste y (a),(b) «cadena no detectada»: `res_spec` queda como candidato; exige corpus real nuevo y un segundo generador.
- Cualquier otra combinación: indeterminada; se documenta y no se interpreta como apoyo.
`lf_eq` se informa, pero no entra en el veredicto (cerrado en §18).

**Predicciones.** (c) artefacto de cadena p≈0,6; (a) cadena detectada p≈0,8; (b) cadena detectada p≈0,6.

**Limitaciones.** Un generador (sdturbo), n=100; R5 aplica el paso bajo a un real ya reducido, lo que no equivale a la historia de un real nativo; la asimetría de escala (reales reducidos, IA nativa) no se rompe aquí; el entrenamiento usa Synthbuster con historial desconocido. Esta prueba no puede demostrar huella de generación, solo descartar o mantener la hipótesis de cadena.

**Resultados:** pendientes (§22).

## 22. Resultados del control de cadena (§21; `e11948a`, `results/h1-chain-run1`)

| rasgo | análisis | run3→dresden | dresden→run3 | media | veredicto |
|---|---|---|---|---|---|
| res_spec | **(c) cadena igualada (primario)** | 0,661 [0,586; 0,739] | 0,608 [0,544; 0,669] | **0,635** | **artefacto de cadena** |
| res_spec | (a) IA propia A5 vs A1 | 0,640 [0,616; 0,673] | 0,643 [0,618; 0,678] | 0,642 | cadena no detectada |
| res_spec | (b) reales R5 vs R1 | 0,904 [0,842; 0,962] | 0,827 [0,790; 0,866] | 0,866 | cadena detectada |
| lf_eq | (c) | 0,242 [0,163; 0,325] | 0,314 [0,257; 0,371] | 0,278 | artefacto de cadena |
| lf_eq | (a) | 0,724 [0,694; 0,761] | 0,688 [0,664; 0,720] | 0,706 | indeterminada |
| lf_eq | (b) | 0,687 [0,642; 0,748] | 0,653 [0,635; 0,681] | 0,670 | cadena no detectada |

**Veredicto preregistrado (§21):** H1-L2-CADENA `res_spec`: **artefacto de cadena** ((c) < 0,68 y (b) «cadena detectada»). `res_spec` queda cerrado como no apoyado.
**Predicciones:** (c) artefacto p≈0,6: cumplida; (b) detectada p≈0,6: cumplida; (a) detectada p≈0,8: **refutada** (0,642).

**Lectura.** El modelo entrenado con la cadena A5 solo en la IA marca como «IA» a fotos reales por aplicarles esa cadena (0,87). Con la cadena en ambos lados la separación cae de 0,94 (§20) a 0,64. Por tanto, el 0,94 de §20 refleja la cadena y no la generación.

**Discordancia (a) (exploratorio, sin probar).** Que A5 vs A1 en la misma IA propia solo dé 0,64 mientras A5/R1 da 0,94 y A1/R1 da 0,51 es compatible con una dispersión por contenido alta en la IA propia, en la que los reales quedan en un rango estrecho; no se ha comprobado.

**Residual no excluido.** (c) 0,61–0,66 con IC inferior > 0,5 no descarta una señal débil, pero está por debajo del umbral preregistrado y no es interpretable como huella de generación. En `lf_eq`, (c) invertido (0,28) confirma que su «se confirma» de §20 era mecánico.

**Erratum sobre §17/§18.** En §18 se llamó «procesado igualado» a la cadena A5 aplicada solo a la IA frente a reales R1; no estaba igualada (los reales no recibían paso bajo ×1,5 ni doble JPEG). La conclusión de §18 «`res_spec` persiste con procesado igualado» queda superada por §22; el texto de §18 no se modifica. Sin explicar queda la separación de Synthbuster sin cadena (A0/R0 0,77), que con IA propia no se reproduce (A1/R1 0,51; A0/R0 0,70 indeterminada): coherente con una propiedad de Synthbuster (historial desconocido y/o asimetría de escala), pero no probada.

**Estado de H1-L2.** `lf_eq`: no apoyada. `res_spec`: no apoyada; no hay evidencia de huella de generación en estos rasgos tras controlar escala, formato, cadena y corpus. Limitaciones: un generador, n=100, reales reducidos, residual débil no excluido.
