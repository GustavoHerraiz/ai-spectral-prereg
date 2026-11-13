# Preregistro H1-R (replicación EXPLORATORIA, procedencia no verificada)

Rama `h1-replicacion`, creada desde el tag `h1-congelada` (`l1-manisha2025` sigue congelada). Fecha: 2026-11-12. Redactado **antes** de ejecutar nada sobre estas carpetas.

## 1. Estatuto
**Exploratorio.** Las carpetas `ai_*` no tienen procedencia documentada en el servidor (sin README, sin script, sin metadatos PNG). No es una replicación confirmatoria y no puede cambiar el veredicto de H1 («no apoyada»). Sirve para ver si el patrón de §22 se repite en otros generadores.

## 2. Procedencia conocida (comprobada el 2026-11-12)
- `ai_sdxl` (50) y `ai_flux_schnell` (50): PNG RGB 512×512, sin metadatos; nombres con índices altos (`sdxl_008675`) y fechas de fichero con 70 ms de diferencia (copia en bloque). Origen, prompts, resolución nativa y reescalado: **desconocidos**.
- `ai_nano_banana_gen_512` / `_raw`: solo este tiene `generation_log.json` (modelo `gemini-3.1-flash-image-preview`, 51 prompts). Los «PNG» raw son JPEG (claves JFIF) de 1408×768; la versión 512×512 es un recorte/reescalado de ese JPEG. **Historial: JPEG previo + reescalado.** `_raw` se excluye (no es 512).
- Resto (`ai_pixart_sigma`, `ai_kandinsky22`, `ai_wuerstchen`, `ai_gpt_image_512`, `ai_nano_banana_512`): procedencia desconocida; se comprobará el tamaño en la ejecución.

## 3. Diseño
Para cada carpeta `ai_*` (excluidas `*_raw`), hasta 100 PNG de 512×512 (los demás tamaños se cuentan y se descartan). Sin modificar los ficheros. Extracción de A1 y A5 con `h1-resample.process` (idéntica a §20). Análisis (a)/(b)/(c) de `h1-chain.py` (§21), con el modelo entrenado en Synthbuster A5 + R5 y los R5/R1 de `h1-chain-run1`. Una fila por carpeta, sin agrupar generadores.
Rasgos: `res_spec` y `lf_eq`. Semilla y bootstrap como en §21.

## 4. Criterios (fijados antes de ver datos)
Los de §21, aplicados por carpeta: (c) media < 0,68 → «artefacto de cadena»; media ≥ 0,75 y mín. IC inf. ≥ 0,60 → «persiste con cadena igualada»; resto «indeterminada». Carpetas con n < 30 imágenes válidas: se informan, **no se interpretan**.
Resumen del conjunto: nº de carpetas en cada categoría. Ninguna categoría se promedia ni se elige a posteriori. Cualquier resultado se informa.

## 5. Predicciones (probabilidad antes de ver datos)
- (c) `res_spec` < 0,68 en la mayoría de las carpetas con n ≥ 30: p ≈ 0,6 (es lo observado con sd-turbo).
- Al menos una carpeta con «persiste con cadena igualada»: p ≈ 0,3. Si ocurre, no se leerá como huella de generación: con procedencia desconocida puede deberse a un historial distinto (reescalado, JPEG previo, formato).
- `ai_nano_banana_gen_512`, con JPEG y reescalado conocidos: (c) distinto del resto, p ≈ 0,5.

## 6. Limitaciones (declaradas de antemano)
Procedencia no verificada; posible reescalado a 512 previo; sin control de contenido ni de prompts; reales siempre reducidos ≥ 3× y IA a 512 (asimetría de escala de §22, sin romper); mismo entrenamiento y corpus que H1; n ≤ 100 por carpeta. Un resultado «también cae» es consistente con §22 pero no lo confirma; uno «persiste» no lo refuta.

## 7. Autotests y ejecución
`h1r-explore.py --self-test` (6 comprobaciones) y `h1-chain.py --self-test` pasan antes de ejecutar. La ejecución no modifica el servidor salvo `results/h1r-explore-run1/`.

## 8. Resultados (ejecutados el 2026-11-12 sobre `ad7e9a5`, `results/h1r-explore-run1`)
8 carpetas, 50 PNG 512×512 cada una (ninguna descartada por tamaño). `res_spec`, media cruzada run3↔dresden:

| carpeta | (c) cadena igualada | (a) A5 vs A1 | veredicto (c) |
|---|---|---|---|
| ai_flux_schnell | 0,472 | 0,964 | artefacto de cadena |
| ai_gpt_image_512 | 0,461 | 0,868 | artefacto de cadena |
| ai_kandinsky22 | 0,525 | 0,948 | artefacto de cadena |
| ai_nano_banana_512 | 0,475 | 0,896 | artefacto de cadena |
| ai_nano_banana_gen_512 | 0,514 | 0,944 | artefacto de cadena |
| ai_pixart_sigma | 0,421 | 0,949 | artefacto de cadena |
| ai_sdxl | 0,559 | 0,845 | artefacto de cadena |
| ai_wuerstchen | 0,429 | 0,898 | artefacto de cadena |

(b) reales R5 vs R1: 0,866 (igual en todas las filas, mismos reales). `lf_eq` (c): 0,433–0,497, todas «artefacto de cadena».

**Predicciones.** (1) Mayoría de (c) `res_spec` < 0,68: **cumplida** (8/8). (2) Alguna carpeta «persiste»: **no ocurrió** (había p≈0,3). (3) `nano_banana_gen_512` distinta del resto: **refutada** (0,514, dentro del rango 0,42–0,56).

**Lectura (exploratoria, procedencia no verificada).**
- Con la cadena igualada, `res_spec` no separa en ninguna de las 8 carpetas; cae a ≈ 0,5 o por debajo. Es consistente con §22 y no lo confirma de forma independiente.
- (a) da 0,85–0,96 en las 8: el modelo marca la cadena A5 en IA. La discordancia de §22(a) (sd-turbo, 0,64) queda como excepción, sin explicar.
- La señal residual de 0,61–0,66 de §22(c) no se reproduce en estas carpetas (0,42–0,56).
- Valores < 0,5 indican que, con cadena igualada, el modelo puntúa estas IA como más «reales» que los reales; no se interpreta como señal.
- La línea de log «VEREDICTO ... (primario c, sdturbo): sin sdturbo» es un defecto cosmético de `h1-chain.py` (nombre de generador fijo); los veredictos por carpeta son los de la tabla.
