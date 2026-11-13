# ¿Separa una huella de alta frecuencia las fotos reales de las imágenes generadas por IA? Un informe negativo preregistrado con controles de procesado

**Versión legible como paper: `PAPER.md`. Este fichero es el registro cronológico (Apéndice A).** Estado: borrador de preprint, informe de resultado negativo, pendiente de revisión externa. Autor: Gustavo Herraiz. Ramas `l1-manisha2025` (congelada, tag `h1-congelada`) y `h1-replicacion` (exploratoria, tag `h1r-resultados`). Los números proceden de los preregistros (`PREREGISTRO_H1.md`, `PREREGISTRO_H1R.md`) y de los resultados crudos versionados en `results-archive/` con SHA-256; las cifras de §5.11 y §5.12 se han contrastado con esos JSON. Este documento no añade resultados: los ordena. Los preregistros están etiquetados en git (`h1-criterios-s*`, `h1r-criterios`) pero no firmados criptográficamente (ver §8).

## Resumen

Se estudia si rasgos espectrales de alta frecuencia (L1-proxy de Manisha 2025, correlaciones de bloques de Mallet 2025, `res_spec` y `lf_eq` propios) separan fotos reales de imágenes de IA más allá del historial de procesado de cada clase. El protocolo fija hipótesis, umbrales y predicciones antes de ver cada resultado; todo lo post hoc se rotula «exploratorio». Resultado hasta la fecha: la separación aparente (AUC 0,8–0,9 en varios corpus) se reduce a ≈ 0,54 al dar el mismo procesado a reales e IA (reescalado a 512 y mismo formato) y no generaliza entre corpus de reales; a resolución nativa con JPEG igualado, `res_spec` sí transfiere entre dos corpus de cámaras (media 0,897), lo que es compatible con una dependencia de la escala del contenido (recorte de foto grande frente a imagen entera). Reconstruir 50 reales con el VAE de Stable Diffusion no las acerca a la clase IA, por lo que el decodificador no explica la separación. **No se afirma detección de origen generativo.** Con un generador propio (sd-turbo, historial conocido) `res_spec` separa a 0,94 solo si la cadena de procesado (paso bajo y doble JPEG) se aplica únicamente a la IA; al aplicar la misma cadena a los reales cae a 0,64 y el modelo marca como IA a reales procesados (0,87). Conclusión: `res_spec` y `lf_eq` no apoyan huella de generación una vez controlados escala, formato, cadena y corpus. `lf_eq` es función de la escala; `res_spec` decae con la reducción (0,90 → 0,78) pero no se hunde, y sigue sin atribuirse a generación por confusiones de remuestreo y de doble compresión (§5.8). Una replicación exploratoria con 8 carpetas de IA de procedencia no verificada da `res_spec` 0,42–0,56 con la cadena igualada (8/8 «artefacto de cadena»). Alcance: rasgos propios y proxy, un benchmark (Synthbuster) y un generador propio; los reales se reducen ≥ 3× mientras la IA es nativa a 512, asimetría de escala que no se ha roto. No es una refutación de los detectores publicados.

## 1. Introducción y objetivo

Los detectores de imágenes sintéticas basados en espectro suelen evaluarse con reales y sintéticas de procedencia distinta (formato, tamaño, historial de compresión). El objetivo es ir más allá de la réplica de Mallet 2025 y Manisha 2025: determinar qué parte de la separación sobrevive cuando se controla el procesado, y publicar también el resultado negativo con el mismo detalle. Preguntas:
- **H1-PRNU:** ¿sirve la presencia del patrón de ruido de sensor (K) del registro como señal de «real»?
- **H1-L2:** ¿separa una huella de baja/alta frecuencia reales de IA de familias no vistas, tras igualar el procesado?

## 2. Metodología general

- **Preregistro previo a los datos:** hipótesis, umbrales, predicciones y desviaciones (registradas antes de ejecutar) en el repositorio.
- **Autotests sintéticos** antes de cada ejecución real: el script debe dar el veredicto esperado con efecto conocido (inyectado) y con efecto nulo.
- **Validación:** dejar una familia de IA fuera (LOFO) y particiones de reales por dispositivo/corpus; IC95 por bootstrap sobre dispositivos (reales) e imágenes (IA); AUC macro por familia.
- **Rotulado:** todo análisis no fijado antes se marca «exploratorio». Las predicciones se registran con probabilidad y se contrastan.
- **Línea base de sesgo:** AUC de bytes (tamaño del archivo) y de bytes/px como control de formato.

## 3. Datos

| Corpus | n | Descripción |
|---|---|---|
| `run3` | 205 reales, 41 dispositivos (≤ 5 por dispositivo) | fotos nativas de móvil, JPEG, lado menor ≥ 2000 px |
| Synthbuster | 700 IA, 7 familias (dalle2, dalle3, firefly, midjourney, sd1, stable-diffusion-2, stable-diffusion-xl) | 100 por familia; PNG salvo firefly (JPEG) |
| Dresden `nat` | 90 reales (≤ 10 por cámara, semilla 20261111) | JPEG de cámara |
| COCO-512 | 50 reales | COCO reescaladas a 512×512 (probablemente Lanczos), PNG; script de preparación no conservado |
| Registro PRNU | 81 K | solo lectura |

El detector v1 está congelado; v3, las 81 K y el código existente son de solo lectura.

## 4. Rasgos

`res_spec`: log-potencia radial del residuo (mediana 3×3, gris), 64 bins logarítmicos en [0,004; 0,5] ciclos/px. `lf_eq`: vector L1-proxy de 32 bins tras ecualizar el procesado (256 + JPEG Q90). Clasificador: estandarización + regresión logística L2 balanceada, LOFO por familia, AUC macro.

## 5. Resultados por etapa (cronológico)

**5.1 Antecedentes (L1, L2, sonda v1).**
- L1 (proxy de Manisha): cerrada en negativo (`PREREGISTRO.md`).
- L2 (Mallet) sobre Synthbuster: confusión de remuestreo y de bytes (AUC de bytes PNG 0,25–0,82; 5/7 familias fuera de [0,40; 0,60]); exploratoria (`PREREGISTRO_L2_MALLET.md`).
- Sonda v1 (`run3`): las dos features congeladas de v1 dan veredicto MIXTO: `jpegPhaseRange` AUC emparejada 0,612 [0,570; 0,655] y `phaseCoherence` 0,611; las medianas por dispositivo real varían más (0,109–0,377) que la diferencia real–IA.
- SD2: propiedad reproducible en tres corpus, causa sin identificar, sin valor como detector.

**5.2 H1-PRNU** (`53cce72`, 995 imágenes, 0 errores). P0 válido (AUC 0,9425). Reales con K: 0,931 [0,905; 0,954]; cámara ausente del registro: 0,587 [0,494; 0,675]; IA sobre umbral: firefly 0,56, resto 0. El 56 % de firefly proviene de un artefacto JPEG genérico (53 de 100 ganadas por una sola K), no de identificación de cámara.

**5.3 H1-L2 inicial.** `lf_eq` AUC macro LOFO 0,799 [0,776; 0,821] («apoyada»); `res_spec` 0,901 [0,885; 0,915] («indeterminada», firefly 0,599). La línea base de bytes separa 0,993.

**5.4 Controles de formato y geometría** (`0799ad1`).

| Brazo | `lf_eq` macro / firefly | `res_spec` macro / firefly |
|---|---|---|
| mod8 | 0,784 / 0,640 | 0,900 / 0,599 |
| q75 | 0,877 / 0,694 | 0,916 / 0,680 |
| q92 | 0,685 / 0,516 | 0,898 / 0,689 |
| down (reales a 1024) | 0,581 / 0,389 | 0,774 / 0,505 |
| down_q92 | 0,581 / 0,353 | 0,815 / 0,721 |

JPEG real frente a firefly (mismo formato): `lf_eq` 0,758, `res_spec` 0,924, bytes/px 1,000. Predicción refutada: «`res_spec` se hunde con el control». Veredicto de controles: parcial para ambos rasgos.

**5.5 Decodificador VAE** (exploratorio, §10 del preregistro; 50 reales, VAE de SD, 512×512).

| Escenario | `res_spec` | `lf_eq` |
|---|---|---|
| A) VAE frente a original | 0,586 [0,536; 0,645] | 0,565 [0,506; 0,631] |
| B) IA frente a VAE, fuera de dominio | 0,613 [0,547; 0,685] | 0,677 [0,594; 0,753] |
| C) IA frente a original, fuera de dominio | 0,624 [0,554; 0,692] | 0,618 [0,540; 0,690] |
| B) en dominio | 0,841 [0,791; 0,888] | 0,662 [0,583; 0,734] |
| C) en dominio | 0,873 [0,826; 0,913] | 0,650 [0,574; 0,727] |

B ≈ C y A ≈ 0,58: el VAE no acerca los reales a la IA, así que el decodificador (este, a 512) no explica la separación. No prueba lo contrario. La caída de 0,87 (dominio) a 0,62 (fuera) apunta a identidad del corpus; «cámaras distintas» es una hipótesis sin comprobar. Limitaciones: un VAE, 50 imágenes, procedencia sin verificar, script de reconstrucción no versionado.

**5.6 Generalización entre corpus (LOCO)** (`ab6e3f3`, 1045 imágenes, pipeline común: recorte cuadrado → Lanczos a 512).

| Brazo / rasgo | Media cruzada (6 pares) | Mín. IC inf. | Veredicto |
|---|---|---|---|
| igual / `res_spec` | 0,536 (pares 0,42–0,62) | 0,353 | artefacto de corpus |
| igual / `lf_eq` | 0,502 (0,44–0,57) | 0,384 | artefacto de corpus |
| nativo / `res_spec` | 0,714 | 0,524 | indeterminada |

Identificación del corpus solo con reales: 0,526 (`res_spec`), 0,453 (`lf_eq`), azar 0,33: «artefacto de corpus» significa aquí «no transfiere», no que el rasgo detecte el corpus. COCO en dominio alcanza 0,897 pero no transfiere (ajuste a un corpus de 50 imágenes).

**5.7 LOCO a resolución nativa** (`e78e3ca`, `run3` ↔ Dresden, JPEG q92 a todas, recorte 512 sin reescalar). `res_spec`: 0,932 [0,903; 0,957] y 0,863 [0,833; 0,891], media 0,897: **aguanta**. `lf_eq`: media 0,733, mín. IC inf. 0,620: indeterminada. Predicción refutada. Interpretación: un recorte de 512 de una foto de ~3000 px es una zona ampliada y suave; la IA a 512 es la escena entera. Ambos corpus «aguantan» entre sí por compartir esa propiedad.

**5.8 Dosis-respuesta de escala** (`fea4c86`, `run3` ↔ Dresden, JPEG q92 a todas). Media cruzada por factor de reducción (1; 1,5; 2; 2,5; 3; P=a 512): `res_spec` 0,897; 0,940; 0,898; 0,842; 0,823; 0,781 → «no explicada por escala» (predicción refutada); `lf_eq` 0,733; 0,733; 0,614; 0,536; 0,520; 0,509 → «dependiente de escala» (cumplida). Discrepancia sin resolver: la §5.6 daba 0,54 para `res_spec` a 512 sin JPEG añadido; aquí 0,78 con JPEG q92 en todas. Confusiones abiertas: paso bajo de Lanczos solo en reales y doble historial JPEG solo en reales.

**5.9 Control de remuestreo y doble compresión** (`48ec33b`). IA con cadenas que imitan al real (paso bajo Lanczos ×1,5, JPEG q90+q92). Comparación decisiva C5 (todo igualado): `res_spec` 0,856 (mín. IC inf. 0,797) → «persiste» (predicción refutada, p≈0,6 de explicada); `lf_eq` 0,631 → «explicada por procesado» (cumplida). Imitar el procesado no reduce `res_spec`: de 0,768 (sin cadena) a 0,856. Discrepancia con §5.6 (0,54): allí la IA también se reducía a 512; hipótesis: la señal está en los píxeles nativos de la IA. Estado: candidato exploratorio; asimetría de escala real/IA estructural; confirmación en corpus nuevo con generador propio pendiente.

**5.10 IA propia de historial conocido** (§19–§20, sd-turbo n=100, 512 nativo, float32+slicing). `res_spec` brazo primario A5/R1: 0,942 [0,907; 0,975 mín.] → «se confirma» (predicción p≈0,4 refutada). Sensibilidad: A1/R1 0,509 (azar, frente a 0,781 de Synthbuster), A0/R0 0,704. `lf_eq`: 0,885 en A5 pero 0,22–0,25 (invertido) sin la cadena. Lectura exploratoria: la separación depende de la cadena de procesado impuesta a la IA, y puede reflejar esa cadena y no la generación; pendiente el control de cadena (§21).

**5.11 Control de cadena** (§21–§22, `e11948a`). Se aplica la cadena A5 también a los reales (R5). Primario (c), IA propia A5 vs R5 con modelo entrenado con la cadena en ambos lados: `res_spec` 0,635 [mín. IC inf. 0,544] → **artefacto de cadena** (predicción p≈0,6 cumplida). Diagnóstico (b): R5 vs R1 = 0,866 → el modelo marca como «IA» a reales por la cadena (p≈0,6 cumplida). Diagnóstico (a): IA propia A5 vs A1 = 0,642, «cadena no detectada» (predicción p≈0,8 refutada; discordancia sin explicar). `lf_eq` (c) 0,278 (invertido). Erratum: la cadena A5 de §5.9 se aplicó solo a la IA y no igualaba el procesado, por lo que su conclusión queda superada.

**5.12 Replicación exploratoria con 8 carpetas de procedencia no verificada** (H1-R, rama `h1-replicacion`, `ad7e9a5`). SDXL, Flux schnell, PixArt-Σ, Kandinsky 2.2, Würstchen, GPT-image y dos de Nano Banana (JPEG previo y reescalado conocidos en este último), 50 PNG de 512×512 cada una. Con la cadena igualada (c), `res_spec` da 0,42–0,56 en las 8 → «artefacto de cadena» en todas; `lf_eq` 0,43–0,50. A5 vs A1 (a): 0,85–0,96 en las 8. Es exploratorio: sin procedencia ni control de prompts, y no confirma §5.11 de forma independiente. La señal residual de 0,61–0,66 no se reproduce; la discordancia (a) de sd-turbo (0,64) queda como excepción sin explicar.

**5.13 Nota exploratoria sobre la discordancia de §5.11(a)** (post hoc, sin preregistro, sobre los JSON archivados). Con el modelo entrenado en Synthbuster A5 frente a R1, la puntuación mediana de sd-turbo ya es «IA» sin la cadena (A1: +1,5 a +2,6) y la cadena solo añade ≈ +1 (A5 vs A1 = 0,64). En las 8 carpetas de §5.12 la cadena aporta ≈ +3 (A1 ≈ reales, −2,7 a −0,6, salvo `nano_banana`, con JPEG de origen) y A5 vs A1 = 0,83–0,98. **Se corrige la hipótesis anterior (dispersión por contenido): el IQR de sd-turbo (2,1–2,3) es similar al de las demás (1,3–2,2); la diferencia es de posición.** sd-turbo es el único generador con algo más que la cadena, coherente con su residual en §5.11(c) (0,635 frente a ≤ 0,56). Causas no distinguibles con estos datos: propiedad de generación de sd-turbo (2 pasos, destilado), el pipeline propio (float32, slicing) o historiales de las otras carpetas que borran el rasgo. Un generador, n=100: es una pista, no un hallazgo; requiere preregistro.

## 6. Discusión

1. **La separación depende del procesado.** Con el mismo reescalado y formato en ambas clases, `res_spec` y `lf_eq` pasan de ≈ 0,8–0,9 a ≈ 0,5–0,6 y no generalizan entre corpus.
2. **A escala nativa `res_spec` sí transfiere entre corpus de cámaras.** Es compatible con una propiedad de la captura (recorte de foto grande frente a imagen completa), consistente con la caída al reducir los reales (§5.4 `down`, §5.6).
3. **No hay evidencia de huella de generación.** El único test dirigido (VAE) descarta que el decodificador sea la causa, sin aportar prueba positiva.
4. **Valor del resultado.** Un negativo bien controlado: los rasgos de tipo Mallet/Manisha se explican por formato, escala y corpus en este benchmark.
5. **`res_spec` era un artefacto de cadena (§5.11).** Su aparente persistencia (§5.9, §5.10) provenía de aplicar el procesado solo a la IA. Con la cadena en ambos lados cae a 0,64 (umbral preregistrado < 0,68).
6. **Lo que el estudio no puede decir.** (i) La asimetría de escala (reales reducidos ≥ 3×, IA nativa a 512) no se rompió: faltan reales de ~512 nativos. (ii) La discordancia de §5.11(a) (sd-turbo A5 vs A1 = 0,64, frente a 0,85–0,96 en las 8 carpetas de §5.12) es un desplazamiento de posición de sd-turbo, no de dispersión (§5.13); su causa sigue sin distinguirse. (iii) Un residual de 0,61–0,66 en §5.11(c) no se reproduce en §5.12 pero no se descarta. (iv) Los resultados se refieren a estos rasgos y este entrenamiento, no a detectores publicados.
7. **Predicciones previas:** refutadas: «`res_spec` se hunde con el control» (§5.4) y «`res_spec`/`lf_eq` artefacto a escala nativa» (§5.7); cumplidas: LOCO igualado (§5.6), cadena (§5.11); también refutada: «el modelo detecta la cadena en la IA propia» (§5.11a).

## 7. Limitaciones

- Un solo generador de IA benchmark (Synthbuster, 7 familias); firefly en JPEG y a 1792 px.
- Reales: 5 fotos por dispositivo; solo 3 corpus; COCO-512 con preparación no documentada (comparabilidad incierta).
- Los reales reducidos llevan huella de paso bajo de Lanczos que las IA no tienen (control en §5.9: no explica `res_spec`).
- Los bootstrap por imagen no modelan la dependencia entre imágenes de un mismo dispositivo/escena cuando cada imagen es su propio clúster (COCO, VAE).
- Los análisis posteriores a la sección 7 del preregistro son exploratorios; solo los de criterios fijados antes (LOCO, nativo, escala) son confirmatorios, y aún con 2–6 pares.
- Un único generador propio (sd-turbo, n=100, float32 con slicing, H1-D3) y Synthbuster como único benchmark; una señal residual débil (0,61–0,66) en §5.11 no se excluye.
- Las 8 carpetas de §5.12 tienen procedencia no verificada y posible reescalado previo; no cuentan como replicación confirmatoria.
- Un VAE de SD a 512; no representa a DALL·E, Midjourney ni Firefly.

## 8. Reproducibilidad

Scripts en `l1-manisha2025/bin/` con autotest: `h1-extract.py`, `h1-analyze.py`, `h1-controls-extract.py`, `h1-post.py`, `h1-vae-recon.py` (no ejecutado), `h1-vae-auc.py`, `h1-loco.py`, `h1-native.py`, `h1-scale.py`, `h1-resample.py`, `h1-owngen-make.py`, `h1-owngen-eval.py`, `h1-chain.py`, `h1r-explore.py`, `archive-results.py`. Resultados crudos en `results-archive/` (24 ficheros, SHA-256 en `MANIFEST.sha256`); tablas consolidadas en `RESULTADOS.md`; orden de criterios/resultados en `CONGELACION.md`; informe en `INFORME_FINAL.md`. Semilla 20261111. Etiquetas git: `h1-criterios-s07…s21`, `h1-resultados-s22`, `h1-informe-final`, `h1-congelada`, `h1r-criterios`, `h1r-resultados`. Las 100 imágenes de sd-turbo se conservan en un tar fuera del servidor (SHA-256 `8c02cf0ea1921c2dd47107ad6afb3a40306ddabed18884b7cb446294c992f9ab`; verificación de la copia pendiente por el autor). Entorno: CPU, Python con `v2_env`; versiones de diffusers y torch en el manifiesto de generación. Pendientes antes de enviar: revisión externa de cifras y texto, un corpus de reales nativos ~512, un segundo generador propio con historial conocido, y documentar el pipeline de COCO-512.

## 9. Criterio para escribir el paper

El criterio se evalúa tras §22: `res_spec` no sobrevive al control de cadena, de modo que no hay candidato que confirmar. El manuscrito es un **informe de resultado negativo**: los rasgos espectrales evaluados no separan origen generativo una vez controlados escala, formato, cadena de procesado y corpus. Reabrir solo con un generador y un corpus nuevos preregistrados.

## Referencias

Mallet et al. 2025 y Manisha et al. 2025 (ver referencias del paper); preregistros del repositorio.
