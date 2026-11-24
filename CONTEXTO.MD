# Contexto del Proyecto: Detección de Granulación en Naranjas Valencia (Multimodal)

## 1. Objetivo del proyecto
Clasificar de forma binaria la granulación interna en naranjas de la variedad Valencia (`0` = Sana, `1` = Granulada) a partir de un dataset experimental de **100 muestras** (`M1` a `M100`), combinando tres modalidades:
1. **Imágenes multicanal** (`X_dataset_multicanal.npy`, tensor `224x224x3` con RGB + UV + NIR).
2. **Espectro NIR del sensor AS7263** (`Dataset_AS7263_M1_M100.xlsx`, mediciones espectrales).
3. **Mediciones de °Brix** (`Mediciones_brix.xlsx`, 4 puntos: `P1`, `P2`, `P3`, `P4`).

## 2. Reglas de trabajo y convenciones
- **No generar gráficos en los scripts de entrenamiento/evaluación de modelos:** la visualización se maneja en scripts aparte dentro de `graficos/`.
- **Consistencia de datos:** Todas las ramas (`NIR MLP`, `CNN`, `Brix MLP` y `Fusión Multimodal`) deben usar la misma partición estratificada guardada en `datos/particiones.npz` (`idx_train`, `idx_val`, `idx_test`).
- **Preprocesamiento sin fuga de datos (*data leakage*):** `StandardScaler` se ajusta (`fit`) **únicamente** sobre `idx_train` y luego se aplica (`transform`) a `val` y `test`. Los pesos de clase (`compute_class_weight("balanced", ...)`) se calculan sobre `y_train`.
- **Nombres de archivos reales:** Priorizar siempre los archivos existentes en `datos/` y la raíz del proyecto (`tensor_multicanal_orange/`), evitando asumir nombres de arreglos `.npy` inexistentes si los datos fuente están en Excel o `.npy` con otro nombre.
- **Carga centralizada y alineada por ID:** Toda carga de datos pasa por `utilidades_datos.py`, que verifica que los IDs `M1`–`M100` coincidan entre etiquetas, NIR, Brix e imágenes (falla si no).
- **Reproducibilidad:** Semilla global `42` (`ud.fijar_semilla()`); dos ejecuciones dan métricas idénticas.
- **Umbral de decisión:** Se elige en **validación** (máx. F1, desempate Youden). Test se reporta con `0.50` y con ese umbral; nunca se ajusta nada con test.
- **Métrica principal:** Validación cruzada estratificada repetida (`evaluar_cv_estratificada.py`, 5 folds × 3). El test fijo (15 muestras, 2 granuladas) es solo referencial por su alta varianza.

## 3. Estructura de carpetas y archivos clave
- `datos/`:
  - `Dataset_AS7263_M1_M100.xlsx`: Datos espectrales NIR (`M1`–`M100`).
  - `Mediciones_brix.xlsx`: Lecturas de °Brix en columnas `P1`, `P2`, `P3`, `P4` (`M1`–`M100`).
  - `Etiquetas_granulacion_M1_M100.xlsx`: Etiquetas binarias de granulación (`0` / `1`).
  - `X_dataset_multicanal.npy`: Tensor de imágenes multicanal (`(100, 224, 224, 3)`).
  - `y_dataset_etiquetas.npy`: Etiquetas en formato `.npy`.
  - `ids_imagenes.npy`: Identificadores de muestras (`M1`–`M100`).
  - `particiones.npz`: Índices fijos de `idx_train`, `idx_val`, `idx_test`.
  - `cvs_nir/` y `fotos_rgb_nir_uv/`: Archivos fuente adicionales.
- `estandarizacion/`:
  - `procesar_espectros_nir.py`: Genera `Dataset_AS7263_M1_M100.xlsx` a partir de los CSVs en `cvs_nir/`.
  - `crear_tensor_multicanal.py`: Genera el tensor multicanal `X_dataset_multicanal.npy` (RGB, UV, NIR) y etiquetas.
- `graficos/`:
  - `utilidades_graficos.py`: Funciones compartidas para curvas de entrenamiento, matrices de confusión, curvas ROC/PR y distribuciones de probabilidad.
  - `graficos_nir.py`: 6 gráficos para la rama NIR MLP (`graficos/graficos_nir_mlp/`).
  - `graficos_brix.py`: 6 gráficos para la rama °Brix MLP (`graficos/graficos_brix_mlp/`).
  - `graficos_cnn.py`: 5 gráficos para la rama CNN multicanal (`graficos/graficos_fotos_cnn/`).
  - `graficos_multimodal.py`: 6 gráficos para el modelo Multimodal y la comparativa global CV (`graficos/graficos_multimodal/`).
  - `visualizar_mallas_roi.py`: Cuadrículas de inspección visual de ROIs por canal (`graficos/inspeccion_roi/`).
  - `analisis_distribucion_tensores.py`: Histogramas, correlaciones y perfiles atípicos del tensor (`graficos/graficos_tensor/`).
  - `graficos_despliegue_app/`: 2 figuras de auditoría operativa y latencia en CPU de `app.py`.
- `modelos_guardados/`: Pesos entrenados (`*.keras`) y escaladores (`*.joblib`). `respaldo_2026-11-22/` conserva los modelos previos a la refactorización.
- `resultados/` (sin gráficos): `metricas_<rama>.json`, `predicciones_test_<rama>.csv`, `historial_<rama>.csv`, los archivos `cv_*.csv` de la validación cruzada y los reportes de trazabilidad `reporte_diagnostico_multimodal_valencia.csv`/`.xlsx`. Son la fuente para los scripts de `graficos/`.
- Módulos compartidos en raíz:
  - `utilidades_datos.py`: Rutas, carga alineada por ID, particiones, semilla, entrenamiento, umbral y métricas.
  - `arquitecturas.py`: Definición de las 4 redes (reutilizada por entrenamiento y CV).
- Scripts principales en raíz:
  - `entrenar_rama_nir.py`: Entrenamiento y evaluación de la rama NIR MLP.
  - `entrenar_rama_cnn.py`: Entrenamiento y evaluación de la rama CNN multicanal.
  - `entrenar_rama_brix.py`: Entrenamiento y evaluación de la rama °Brix MLP.
  - `entrenar_multimodal.py`: Entrenamiento y evaluación de la red conjunta multimodal.
  - `evaluar_cv_estratificada.py`: Validación cruzada estratificada repetida de todas las ramas + línea base de regresión logística.
  - `run_pipeline.py`: Orquestador (`--con-preprocesamiento`, `--con-cv`).
  - `test_cargar_modelos.py`: Evalúa cada modelo + escalador con los datos reales de test y verifica que coincidan con `resultados/`.
  - `app.py`: Interfaz gráfica web local interactiva en Streamlit (`streamlit run app.py`, inferencia en ~26–28 ms en CPU).
- `requirements.txt`: Versiones exactas del entorno.
- `README.md`: Documentación pública completa para GitHub.

## 4. Modelos, artefactos y resultados reproducibles (`semilla=42`)

### 4.1 Evaluación en Validación Cruzada Estratificada Repetida (5 folds × 3 repeticiones, `n=100` OOF)
> **Métrica oficial y confiable** (reportada como `media ± desv. est.` entre las 3 repeticiones con umbral elegido en validación interna; sin fuga de datos):

| Rama / Modelo | ROC-AUC | F1 (Granulada) | Recall (Granulada) | Especificidad | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fusión Multimodal** (`entrenar_multimodal.py`) | **0.613 ± 0.119** | **0.292 ± 0.080** | **0.722 ± 0.193** | **0.557 ± 0.080** | **0.639 ± 0.104** |
| **Rama °Brix MLP** (`entrenar_rama_brix.py`) | 0.574 ± 0.094 | 0.223 ± 0.018 | 0.583 ± 0.144 | 0.496 ± 0.184 | 0.540 ± 0.036 |
| **Rama NIR MLP** (`entrenar_rama_nir.py`) | 0.514 ± 0.038 | 0.235 ± 0.018 | 0.611 ± 0.241 | 0.508 ± 0.259 | 0.559 ± 0.034 |
| **Rama CNN Multicanal** (`entrenar_rama_cnn.py`) | 0.485 ± 0.049 | 0.147 ± 0.024 | 0.389 ± 0.048 | 0.466 ± 0.039 | 0.427 ± 0.044 |
| *Baseline LogReg NIR* | 0.522 ± 0.043 | 0.204 ± 0.024 | 0.611 ± 0.193 | 0.409 ± 0.172 | 0.510 ± 0.038 |
| *Baseline LogReg °Brix* | 0.515 ± 0.068 | 0.212 ± 0.022 | 0.722 ± 0.210 | 0.318 ± 0.149 | 0.520 ± 0.030 |

### 4.2 Artefactos guardados (entrenados sobre `datos/particiones.npz`, `semilla=42`)
- **Rama NIR (`entrenar_rama_nir.py`):**
  - MLP funcional con salida sigmoide (`salida_nir`) y capa latente `feat_nir`.
  - Artefactos en `modelos_guardados/`: `modelo_nir.keras`, `scaler_nir.joblib`.
  - Test fijo (`n=15`, 13 sanas / 2 granuladas): `ROC-AUC = 0.5769` | Umbral `0.50`: `F1 = 0.2500`, `Recall = 0.5000`, `Acc = 0.6000` | Umbral val (`0.634`): `F1 = 0.0000`, `Acc = 0.8667`.
- **Rama CNN (`entrenar_rama_cnn.py`):**
  - CNN sobre `(224, 224, 3)` con aumento suave (`RandomFlip`, `RandomRotation`), bloques `Conv2D` + `MaxPooling2D`, `GlobalAveragePooling2D(name="feat_cnn")`, `Dense(64)`, `Dropout(0.5)` y salida sigmoide (`salida_cnn`).
  - Artefacto en `modelos_guardados/`: `modelo_cnn.keras`.
  - Test fijo (`n=15`): `ROC-AUC = 0.7308` | Umbral `0.50`: `F1 = 0.0000`, `Acc = 0.8667` | Umbral val (`0.496`): `F1 = 0.2222`, `Recall = 0.5000`, `Acc = 0.5333`.
- **Rama Brix (`entrenar_rama_brix.py`):**
  - MLP sobre 4 entradas (`P1`–`P4`): `Dense(16)` -> `Dense(8, name="feat_brix")` -> `Dropout(0.2)` -> salida sigmoide (`salida_brix`).
  - Artefactos en `modelos_guardados/`: `modelo_brix.keras`, `scaler_brix.joblib`.
  - Test fijo (`n=15`): `ROC-AUC = 0.6154` | Umbral `0.50`: `F1 = 0.3333`, `Recall = 1.0000`, `Acc = 0.4667` | Umbral val (`0.528`): `F1 = 0.0000`, `Acc = 0.7333`.
- **Modelo Multimodal (`entrenar_multimodal.py`):**
  - Arquitectura conjunta con 3 entradas (`in_imagen`, `in_nir`, `in_brix`), aumento suave y regularización (`Dropout`) por rama, concatenación (`Concatenate`) de características y clasificación binaria sigmoide (`salida_multimodal`).
  - Artefactos en `modelos_guardados/`: `modelo_multimodal.keras`, `scaler_multimodal_nir.joblib`, `scaler_multimodal_brix.joblib`.
  - Test fijo (`n=15`): `ROC-AUC = 0.5385` | Umbral `0.50`: `F1 = 0.0000`, `Acc = 0.6000` | Umbral val (`0.703`): `F1 = 0.0000`, `Acc = 0.7333`.

