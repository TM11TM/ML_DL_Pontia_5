# Informe DL ML

**Definición de los roles de la pareja (quién hace qué) 🤝**

Los roles dentro de nuestro equipo son los siguientes:

- **Amalia Martin:**
  - Modelos :
  - Evaluación :
  - Automatización : ...
- **Carlos Toro:**
  - Modelos : regresión lineal, árbol de decisión, Random Forest, XGBoost
  - Evaluación : regresión lineal, árbol de decisión, Random Forest, XGBoost
  - Automatización : Division en scripts

Ambos nos hemos dividido de manera que cada uno pueda abordar todos los aspectos del proyecto, para asegurarnos de aplicar todos los conceptos que hemos aprendido en clase.

**Justificación del problema 🎯**

El problema que estamos abordando con este dataset es calcular cuántas personas cancelan sus reservas en un hotel. Este dataset contiene todos los datos necesarios para analizar y determinar las razones por las cuales alguien decide cancelar su reserva.

**Análisis exploratorio de datos 📊**
EDA ...

**Diseño del sistema 🖥️**

El proyecto está estructurado de la siguiente manera:

```pyhthon
ML_DL_PONTIA_5/
	│
	├── /data
	│   ├── dataset_hotel_preprocessed.csv
		├── dataset_practica_final
		└── resultado_modelos
	├── /env-entrega-final                        # Entorno virtual
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
	├── /models
	│   └── modelo_random_forest.pkl
	├── requirements.txt
	├── Informe Final
	└── README.md
```

**Resultados y elección final 🌟**
Después de probar los diferentes modelos hemos decidido que el mejor es el modelo de **XGBoost**

### 📊 **Comparativa rápida de los modelos:**

| Modelo                 | Accuracy  | Precisión | Recall    | F1-Score  | AUC       |
| ---------------------- | --------- | --------- | --------- | --------- | --------- |
| Regresión Logística    | 0.758     | 0.655     | 0.254     | 0.366     | 0.756     |
| Árbol de Decisión      | 0.757     | 0.638     | 0.253     | 0.362     | 0.757     |
| Random Forest          | 0.757     | 0.638     | 0.253     | 0.362     | 0.757     |
| **XGBoost**            | **0.783** | **0.681** | **0.399** | **0.503** | **0.802** |
| Red Neuronal Multicapa | 0.780     | 0.676     | 0.386     | 0.491     | 0.802     |

**¿Por qué hemos elegido XGBoost?**

La métrica mas determinante de nuestros modelos es el F1-score, al ser la media de recall (el cual es alto, lo que nos permite identificar un mayor porcentaje de reservas que se cancelarán) y precisión. XGBoost tiene además mayor accuracy y AUC a parte del F1-score, lo cual hace que sean las mejores métricas obtenidas de todos los modelos.

Aunque la red neuronal multicapa también tiene métricas buenas, XGBoost las supera y además suele ser mas rápido de entrenar y ajustar que la red neuronal.

En resumen XGBoost balancea bien la detección de cancelaciones que si son cancelaciones (cancelaciones reales) sin generar excesivos errores, lo cual a un hotel o cadena de hoteles le permite reducir mucho las cancelaciones.

**Reflexión crítica sobre limitaciones y mejoras 🤔**

Hemos identificado que nuestro modelo ofrece buenos resultados, pero son mejorables, ya que tenemos algunas métricas que no están en sus mejores valores, por ejemplo el recall, es decir, que no detectamos todas las cancelaciones reales y eso puede afectar a la capacidad del hotel para detectar todas las cancelaciones.

Como mejoras futuras, deberíamos de probar mas técnicas de balanceo de clases mejores y hacer otro tipo de pruebas mas avanzadas. Esto nos ayudaría a predecir mejor y diseñar estrategias mas fiables para reducir la tasa de cancelación de nuestro hotel.
