# 🤖 ML_DL_Pontia_5
# 📊 Proyecto de Clasificación Binaria

## 👥 Autores
-  Amalia Martín  
-  Carlos Toro

---

## 🎯 Descripción del Problema y Datos

### ✅ Objetivos
El objetivo de esta práctica es implementar todo lo aprendido a lo largo del módulo. Para ello, diseñamos e implementamos un sistema automático que:

- 🔁 Entrene, evalúe y compare distintos modelos de **clasificación binaria**.
- 🏆 Seleccione el mejor modelo según una **métrica principal**, mostrando también otras secundarias.
- ⚙️ Automatice el flujo completo desde los datos hasta la inferencia.

### 🧾 Datos
- 📁 Origen de los datos: archivo CSV proporcionado → `data/dataset_practica_final`
- 📊 Descripción de características: (se detallan en el informe).
- 🎯 Variable objetivo: Binaria (`0` o `1`) — representa la clase a predecir.

---

## 🛠️ Instrucciones para Ejecutar el Proyecto

###  Requisitos
- 🐍 Python versión 3.9
- 📦 Librerías necesarias (ver `requirements.txt`)

### Pasos para ejecución

1. **📥 Clonar el repositorio:**
   ```bash
   git clone https://github.com/usuario/repositorio.git
   cd repositorio
   ```

2. **🐍 Crear y activar entorno virtual (opcional pero recomendado):**
   ```powershell
   python -m venv env-entrega-final
   .\env-entrega-final\Scripts\Activate.ps1
   ```

3. **📦 Instalar dependencias:**
   ```powershell
   pip install -r requirements.txt
   ```


---

## 📂 Estructura del Proyecto

```
ML_DL_PONTIA_5/
	│
	├── /data
	│   ├── dataset_hotel_preprocessed.csv
		├── dataset_practica_final
		└── resultado_modelos
	├── /notebooks
		├──  dl_ml.ipynb
		└──  EDA.ipynb
	├──	/process_hotel_model
		├── config.py
		├── trainer.py
		├── data_loader.py
		├── preprocess.py
		├── model.py
		├── metrics.py
	├── requirements.txt
	├── Informe Final.md
	└── README.md
```

---

## ⚙️ Flujo del Proyecto

1. **Preprocesamiento:**
   - Limpieza y transformación de datos (`preprocess.py`, `data_loader.py`).
2. **Entrenamiento y evaluación:**
   - Entrena varios modelos de clasificación binaria (`model.py`, `trainer.py`).
   - Evalúa con métricas como accuracy, F1, ROC-AUC (`metrics.py`).
3. **Selección de modelo:**
   - Compara resultados y selecciona el mejor modelo según la métrica principal.
4. **Resultados:**
   - Guarda métricas y predicciones en `data/resultados_modelos.csv`.
   - Visualizaciones en los notebooks y en `notebooks/iframe_figures/`.

---

## 🧪 Modelos y Métricas

- **Modelos implementados:**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Redes Neuronales (MLP)
- **Métricas principales:**
  - Accuracy
  - F1 Score
  - ROC-AUC
  - Precision, Recall

---

## 📈 Resultados

- Los resultados de cada modelo se guardan en `data/resultados_modelos.csv`.
- Las figuras y visualizaciones se encuentran en `notebooks/iframe_figures/`.
- El informe final (`Informe final.md`) contiene el análisis detallado y la justificación de la selección del modelo.

---

## 🛠️ Personalización

- Para modificar parámetros de modelos, editar `process_hotel_model/config.py`.
- Para añadir nuevos modelos, crear el script correspondiente en `process_hotel_model/model.py` y actualizar el pipeline en `trainer.py`.
- Para cambiar la métrica principal, modificar la selección en `trainer.py`.

---
