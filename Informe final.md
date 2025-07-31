# Informe DL ML

**Definición de los roles de la pareja (quién hace qué) 🤝**

Los roles dentro de nuestro equipo son los siguientes:

- **Amalia Martin:**
  - Modelos : red neuronal multicapa
  - Evaluación : red neuronal multicapa
  - Desarrollo : análisis exploratorio de datos

- **Carlos Toro:**
  - Modelos : regresión lineal, árbol de decisión, Random Forest, XGBoost
  - Evaluación : regresión lineal, árbol de decisión, Random Forest, XGBoost
  - Automatización : Division en scripts

Ambos nos hemos dividido de manera que cada uno pueda abordar todos los aspectos del proyecto, para asegurarnos de aplicar todos los conceptos que hemos aprendido en clase.

Además, durante una llamada poníamos en común la parte que había hecho cada uno para que quedara claro a la otra parte la forma en la que se había hecho y por qué.

**Justificación del problema 🎯**

El problema que estamos abordando con este dataset es calcular cuántas personas cancelan sus reservas en un hotel. Este dataset contiene todos los datos necesarios para analizar y determinar las razones por las cuales alguien decide cancelar su reserva.

**Análisis exploratorio de datos 📊**

El análisis exploratorio de datos (EDA) realizado sobre el dataset aportado ha permitido comprender la estructura y calidad de los datos, así como identificar patrones relevantes para la modelización.

Se han realizado varias fases diferenciadas: primero se realizó la carga y revisión inicial de los datos, seguida del tratamiento de valores nulos y la reducción de variables irrelevantes. Posteriormente se analizó la variable objetivo y se clasificaron los tipos de variables presentes. Se identificaron y trataron outliers, se estudió la distribución y correlación de las variables, y finalmente se guardó el dataset procesado para su uso en el modelado.

El tratamiento de los datos realizado en base a las conclusiones vistas en el análisis ha permitido preparar un conjunto de datos robusto y adecuado para la aplicación de modelos de clasificación binaria, asegurando la calidad y relevancia de las variables seleccionadas.

**Resultados y elección final 🌟**
Después de probar los diferentes modelos hemos decidido que el mejor es el modelo de **XGBoost**

### 📊 **Comparativa rápida de los modelos:**

| Modelo                 | Accuracy  | Precisión | Recall    | F1-Score  | AUC       |
| ---------------------- | --------- | --------- | --------- | --------- | --------- |
| Regresión Logística    | 0.758     | 0.655     | 0.254     | 0.366     | 0.756     |
| Árbol de Decisión      | 0.703     | 0.468     | 0.473     | 0.470     | 0.634     |
| Random Forest          | 0.757     | 0.581     | 0.443     | 0.503     | 0.757     |
| **XGBoost**            | **0.783** | **0.666** | **0.419** | **0.515** | **0.802** |
| Red Neuronal Multicapa | 0.780     | 0.676     | 0.386     | 0.491     | 0.802     |

**¿Por qué hemos elegido XGBoost?**

La métrica más determinante de nuestros modelos es el F1-score, al ser la media de recall (el cual es alto, lo que nos permite identificar un mayor porcentaje de reservas que se cancelarán) y precisión. XGBoost tiene además mayor accuracy y AUC a parte del F1-score, lo cual hace que sean las mejores métricas obtenidas de todos los modelos.

Aunque la red neuronal multicapa también tiene métricas buenas, XGBoost las supera y además suele ser más rápido de entrenar y ajustar que la red neuronal.

En resumen XGBoost balancea bien la detección de cancelaciones que si son cancelaciones (cancelaciones reales) sin generar excesivos errores, lo cual a un hotel o cadena de hoteles le permite reducir mucho las cancelaciones.

**Reflexión crítica sobre limitaciones y mejoras 🤔**

Hemos identificado que nuestro modelo ofrece buenos resultados, pero son mejorables, ya que tenemos algunas métricas que no están en sus mejores valores, por ejemplo el recall, es decir, que no detectamos todas las cancelaciones reales y eso puede afectar a la capacidad del hotel para detectar todas las cancelaciones.

Como mejoras futuras, deberíamos de probar mas técnicas de balanceo de clases mejores y hacer otro tipo de pruebas mas avanzadas. Esto nos ayudaría a predecir mejor y diseñar estrategias mas fiables para reducir la tasa de cancelación de nuestro hotel.
