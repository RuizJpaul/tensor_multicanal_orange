# Detección de Granulación Interna en Naranjas Valencia mediante Fusión Multimodal

Sistema de aprendizaje profundo y aprendizaje estadístico para la clasificación binaria de granulación interna (`0` = Sana, `1` = Granulada) en naranjas de la variedad **Valencia** (*Citrus sinensis* L. Osbeck) a partir de un dataset experimental de **100 muestras (`M1`–`M100`)**, integrando tres modalidades complementarias:

1. **Visión Multicanal (`224 × 224 × 3`):** Tensor de imágenes construido a partir de recortes automáticos de región de interés (ROI) en los espectros **Visible (RGB)**, **Ultravioleta (UV)** e **Infrarrojo Cercano (NIR)**.
2. **Espectroscopía NIR de Bajo Costo (Sensor AS7263):** 24 variables espectrales correspondientes a 6 longitudes de onda (`R: 610 nm`, `S: 680 nm`, `T: 730 nm`, `U: 760 nm`, `V: 810 nm`, `W: 860 nm`) medidas en 4 puntos ecuatoriales (`P1`–`P4`).
3. **Sólidos Solubles Totales (°Brix):** 4 mediciones refractométricas ecuatoriales (`P1`, `P2`, `P3`, `P4`) por fruto.

---

## Estructura del Repositorio

```text
tensor_multicanal_orange/
├── datos/
│   ├── cvs_nir/                             # 100 archivos CSV crudos del sensor AS7263 (M1.csv - M100.csv)
│   ├── fotos_rgb_nir_uv/                    # 300 imágenes crudas (100 muestras × 3 canales: RGB, UV, N)
│   ├── Dataset_AS7263_M1_M100.xlsx          # Matriz espectral NIR consolidada (100 × 24 variables)
│   ├── Mediciones_brix.xlsx                 # Lecturas de °Brix en P1, P2, P3 y P4
│   ├── Etiquetas_granulacion_M1_M100.xlsx   # Diagnóstico binario (0 = Sana, 1 = Granulada) y confianza
│   ├── X_dataset_multicanal.npy             # Tensor normalizado de imágenes (100, 224, 224, 3) en float32
│   ├── y_dataset_etiquetas.npy              # Vector de etiquetas (100,) en int32
│   ├── ids_imagenes.npy                     # Identificadores de muestra ('M1'..'M100')
│   └── particiones.npz                      # Partición estratificada fija (idx_train=69, idx_val=16, idx_test=15)
│
├── estandarizacion/
│   ├── procesar_espectros_nir.py            # Limpia y promedia los CSV de cvs_nir/ -> Dataset_AS7263_M1_M100.xlsx
│   └── crear_tensor_multicanal.py           # Extrae ROI por contorno y genera X_dataset_multicanal.npy
│
├── modelos_guardados/
│   ├── modelo_nir.keras                     # MLP entrenado para la rama NIR
│   ├── scaler_nir.joblib                    # StandardScaler ajustado en train para NIR unimodal
│   ├── modelo_brix.keras                    # MLP entrenado para la rama °Brix
│   ├── scaler_brix.joblib                   # StandardScaler ajustado en train para °Brix unimodal
│   ├── modelo_cnn.keras                     # CNN entrenada sobre el tensor multicanal (224, 224, 3)
│   ├── modelo_multimodal.keras              # Red neuronal de fusión intermedia (Imagen + NIR + °Brix)
│   ├── scaler_multimodal_nir.joblib         # StandardScaler NIR del modelo multimodal
│   └── scaler_multimodal_brix.joblib        # StandardScaler °Brix del modelo multimodal
│
├── resultados/
│   ├── cv_resumen.csv                                     # Tabla resumen (media ± DE) de la Validación Cruzada 5×3
│   ├── cv_oof_por_repeticion.csv                          # Métricas out-of-fold completas (n=100) por repetición
│   ├── cv_por_fold.csv                                    # Métricas detalladas por cada uno de los 15 folds
│   ├── cv_predicciones_<rama>.csv                         # Probabilidades predichas OOF por muestra y repetición
│   ├── metricas_<rama>.json                               # Métricas en validación y test de la partición fija
│   ├── predicciones_test_<rama>.csv                       # Predicciones individuales en el conjunto de test (n=15)
│   ├── historial_<rama>.csv                               # Curvas de aprendizaje época por época (loss, auc, accuracy)
│   ├── reporte_diagnostico_multimodal_valencia.csv        # Reporte de trazabilidad RF-10 exportado desde app.py (.CSV)
│   └── reporte_diagnostico_multimodal_valencia.xlsx       # Reporte de trazabilidad RF-10 exportado desde app.py (.XLSX)
│
├── graficos/
│   ├── utilidades_graficos.py               # Funciones compartidas de visualización científica (200 DPI)
│   ├── graficos_nir.py                      # Genera 6 figuras de la rama NIR en graficos/graficos_nir_mlp/
│   ├── graficos_brix.py                     # Genera 6 figuras de la rama °Brix en graficos/graficos_brix_mlp/
│   ├── graficos_cnn.py                      # Genera 5 figuras de la rama CNN en graficos/graficos_fotos_cnn/
│   ├── graficos_multimodal.py               # Genera 6 figuras multimodales y comparativas en graficos/graficos_multimodal/
│   ├── analisis_distribucion_tensores.py    # Genera 5 figuras exploratorias en graficos/graficos_tensor/
│   ├── visualizar_mallas_roi.py             # Genera 12 mallas de inspección ROI en graficos/inspeccion_roi/
│   └── graficos_despliegue_app/             # 2 figuras de auditoría visual y latencia en CPU de app.py (Streamlit)
│
├── utilidades_datos.py                      # Carga verificada por ID, semilla determinista, entrenamiento y métricas
├── arquitecturas.py                         # Definición centralizada de las arquitecturas Keras (MLP, CNN y Multimodal)
├── entrenar_rama_nir.py                     # Entrenamiento y evaluación de la rama espectral NIR
├── entrenar_rama_brix.py                    # Entrenamiento y evaluación de la rama °Brix
├── entrenar_rama_cnn.py                     # Entrenamiento y evaluación de la rama CNN multicanal
├── entrenar_multimodal.py                   # Entrenamiento y evaluación del modelo de Fusión Multimodal
├── evaluar_cv_estratificada.py              # Validación cruzada estratificada repetida (5 folds × 3 repeticiones)
├── test_cargar_modelos.py                   # Prueba automatizada de carga e inferencia sobre datos reales de test
├── run_pipeline.py                          # Orquestador completo del proyecto
├── app.py                                   # Aplicación web local interactiva en Streamlit (< 50 ms en CPU)
├── requirements.txt                         # Dependencias con versiones exactas
└── CONTEXTO.MD                              # Documentación técnica interna del proyecto
```

---

## Metodología y Rigor Experimental

- **Verificación estricta de alineación por ID:** [`utilidades_datos.py`](utilidades_datos.py) comprueba explícitamente que el orden `M1`–`M100` coincida fila por fila entre etiquetas, espectro NIR, mediciones °Brix y tensores de imagen antes de entrenar.
- **Cero fuga de datos (*No Data Leakage*):**
  - El escalado (`StandardScaler`) se ajusta (`fit`) **únicamente** sobre el subconjunto de entrenamiento (`idx_train`) y se aplica (`transform`) a validación y test.
  - Los pesos de compensación de desbalance (`compute_class_weight("balanced", ...)`) se calculan exclusivamente sobre `y_train`.
  - El umbral óptimo de clasificación se calibra maximizando el *F1-score* (con desempate por índice de Youden) **únicamente sobre el conjunto de validación**, nunca sobre test.
- **Reproducibilidad determinista:** Se fija la semilla global (`SEMILLA = 42`) en Python, NumPy y TensorFlow/Keras con `TF_ENABLE_ONEDNN_OPTS=0` y `enable_op_determinism()`.
- **Validación Cruzada Estratificada Repetida ($5 \text{ folds} \times 3 \text{ repeticiones}$):** Dado el tamaño del dataset ($N = 100$, con $88$ frutos sanos y $12$ granulados), la comparación principal entre modelos se realiza mediante predicciones *out-of-fold* (OOF) sobre la totalidad de las muestras a lo largo de 3 repeticiones independientes.

---

## Resultados Principales (Validación Cruzada Estratificada 5×3, $N = 100$)

Métricas *out-of-fold* reportadas como **`media ± desviación estándar`** entre repeticiones con umbral calibrado en validación interna (fuente: [`resultados/cv_resumen.csv`](resultados/cv_resumen.csv)):

| Modelo / Rama | ROC-AUC | F1 (Granulada) | Sensibilidad / Recall | Especificidad | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fusión Multimodal (Imagen + NIR + °Brix)** | **0.613 ± 0.119** | **0.292 ± 0.080** | **0.722 ± 0.193** | **0.557 ± 0.080** | **0.639 ± 0.104** |
| **Rama °Brix MLP** | 0.574 ± 0.094 | 0.223 ± 0.018 | 0.583 ± 0.144 | 0.496 ± 0.184 | 0.540 ± 0.036 |
| **Rama NIR MLP (AS7263)** | 0.514 ± 0.038 | 0.235 ± 0.018 | 0.611 ± 0.241 | 0.508 ± 0.259 | 0.559 ± 0.034 |
| **Rama CNN Multicanal (RGB+UV+NIR)** | 0.485 ± 0.049 | 0.147 ± 0.024 | 0.389 ± 0.048 | 0.466 ± 0.039 | 0.427 ± 0.044 |
| *Baseline Regresión Logística NIR* | 0.522 ± 0.043 | 0.204 ± 0.024 | 0.611 ± 0.193 | 0.409 ± 0.172 | 0.510 ± 0.038 |
| *Baseline Regresión Logística °Brix* | 0.515 ± 0.068 | 0.212 ± 0.022 | 0.722 ± 0.210 | 0.318 ± 0.149 | 0.520 ± 0.030 |

> La **Fusión Multimodal** supera de forma consistente a cada modalidad individual tanto en discriminación global (*ROC-AUC* = `0.613`, *Balanced Accuracy* = `63.9%`) como en detección de frutos afectados (*Recall* = `72.2%`, *F1* = `0.292`).

---

## Arquitecturas Implementadas y Espacio Latente (`arquitecturas.py`)

| Componente / Modelo | Artefacto en `modelos_guardados/` | Entrada | Vector Latente | Parámetros | Umbral Calibrado ($\tau^*$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. Rama Visual CNN** | `modelo_cnn.keras` | Tensor `(224, 224, 3)` RGB+UV+NIR | `feat_cnn` (`128` dim) | `101 569` | `0.496` |
| **2. Rama Espectral NIR MLP** | `modelo_nir.keras` + `scaler_nir.joblib` | `24` variables AS7263 (`610–860 nm`) | `feat_nir` (`32` dim) | `3 713` | `0.634` |
| **3. Rama Fisicoquímica °Brix MLP** | `modelo_brix.keras` + `scaler_brix.joblib` | `4` puntos ecuatoriales (`P1–P4`) | `feat_brix` (`8` dim) | `225` | `0.528` |
| **4. Fusión Multimodal** | `modelo_multimodal.keras` + 2 scalers | Imagen + NIR (`24`) + °Brix (`4`) | `capa_fusion` (**`168` dim**) | `110 073` | `0.703` |

---

## Interfaz Gráfica Web Local en Streamlit (`app.py`)

El módulo [`app.py`](app.py) despliega localmente los cuatro modelos entrenados sobre CPU convencional (con precalentamiento de grafo `@st.cache_resource` y tiempos de respuesta promedio de **~26–28 ms por fruto**, muy por debajo del límite operativo de `200 ms`):

- **Modo 1 — Explorador y Auditoría del Lote Experimental (`M1`–`M100`):** Permite filtrar por partición (*Prueba Independiente 15 %*, *Validación 16 %*, *Entrenamiento 69 %* o *Lote completo*) y condición real (*Sanas* vs. *Granuladas*), visualizando en tiempo real el tensor `224×224×3` (RGB, UV, NIR y falso color), la curva espectral AS7263 (`610–860 nm`), el perfil ecuatorial de `°Brix` y las probabilidades comparadas de las 4 arquitecturas frente al corte real (*Ground Truth*).
- **Modo 2 — Diagnóstico de Nueva Muestra en Laboratorio:** Incluye validadores de entrada (`RNF-04`) para subir fotografías crudas (`RGB`, `UV`, `NIR`) con recorte automático de ROI, procesar archivos `.csv` exportados desde la interfaz Web Serial del sensor AS7263 e ingresar lecturas de `°Brix`.
- **Historial y Exportación Tabular (`RF-10`):** Registra cada evaluación de la sesión y permite descargar reportes auditables en formatos **`.CSV`** y **`.XLSX` (Excel)**.

---

## Instalación y Uso

### 1. Crear entorno virtual e instalar dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2. Ejecutar el pipeline de entrenamiento y verificación

```powershell
# Entrena las 4 ramas sobre datos/particiones.npz y verifica los artefactos guardados
python run_pipeline.py

# (Opcional) Regenerar primero los datasets desde los CSV e imágenes crudas
python run_pipeline.py --con-preprocesamiento

# (Opcional) Ejecutar también la validación cruzada estratificada 5x3 completa
python run_pipeline.py --con-cv
```

### 3. Ejecutar scripts individuales

```powershell
# Entrenar modelos por separado
python entrenar_rama_nir.py
python entrenar_rama_brix.py
python entrenar_rama_cnn.py
python entrenar_multimodal.py

# Validación cruzada estratificada repetida (todas las ramas o un subconjunto)
python evaluar_cv_estratificada.py
python evaluar_cv_estratificada.py --ramas nir brix

# Verificar integridad y reproducción exacta de métricas de los modelos guardados
python test_cargar_modelos.py
```

### 4. Generar todas las figuras de análisis y resultados

```powershell
python graficos/graficos_nir.py
python graficos/graficos_brix.py
python graficos/graficos_cnn.py
python graficos/graficos_multimodal.py
python graficos/analisis_distribucion_tensores.py
python graficos/visualizar_mallas_roi.py
```

### 5. Ejecutar la interfaz gráfica local en Streamlit

```powershell
streamlit run app.py
```
