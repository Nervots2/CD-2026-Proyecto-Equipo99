# Adicción a la Pantalla

**Detección de uso problemático de pantallas en jóvenes adultos**

Proyecto incremental de Ciencia de Datos 2026 · Universidad Técnica Federico Santa María · Prof. Jesús A. Parra

**Equipo 99:** Javier Andrade · Lucas Martínez · Benjamín Ramos

## Problema y pregunta

El uso intensivo de pantallas suele asociarse con menos horas de sueño y mayor estrés. Sin embargo, distinguir un uso alto de un uso problemático (adicción) sigue haciéndose "a ojo", porque no sabemos qué hábitos concretos marcan la diferencia. Nos enfocamos en personas de 18 a 35 años, donde una intervención de higiene digital todavía puede cambiar hábitos.

**Pregunta del proyecto:**

> ¿Es posible identificar si una persona de 18 a 35 años presenta un patrón de uso problemático de pantallas (Y), a partir de sus hábitos digitales, de sueño y de su contexto personal medidos en una semana típica (X), usando únicamente la información disponible al momento de la evaluación (T)?

## Definición de X, Y y T

| Componente | Definición |
| :--- | :--- |
| **X (predictores)** | 9 variables numéricas de conducta digital y sueño, y 3 categóricas de contexto personal (ver la tabla de variables). |
| **Y (objetivo)** | `addicted_label`: indicador binario de uso problemático de pantallas (1 = sí presenta, 0 = no presenta). Es un problema de clasificación. |
| **T (contexto)** | Una semana típica de uso, con contraste entre días hábiles y fin de semana. La evaluación es transversal: describe el estado actual de la persona, sin serie de tiempo ni proyección a futuro. |

## Dataset

- **Fuente:** competición pública de Kaggle sobre hábitos de uso de pantalla.
- **Unidad de observación:** una persona por fila.
- **Tamaño:** 691.369 registros y 14 variables (1 identificador, 12 predictores y 1 variable objetivo).
- **Variable objetivo:** 70,9 % de las personas con uso problemático y 29,1 % sin uso problemático.
- **Calidad:** sin duplicados ni valores fuera de rango. Cada predictor tiene entre 4 % y 19 % de valores faltantes, y solo el 38,9 % de las filas está completa.

La competición incluía además 296.302 registros sin etiqueta. Se descartaron porque no permiten evaluar un modelo, así que el proyecto trabaja solo con los registros etiquetados.

| Variable | Descripción | Tipo | Rol |
| :--- | :--- | :--- | :--- |
| `daily_screen_time_hours` | Horas de pantalla en un día hábil | Numérica | Predictor (X) |
| `weekend_screen_time` | Horas de pantalla en un día de fin de semana | Numérica | Predictor (X) · contexto T |
| `social_media_hours` | Horas diarias en redes sociales | Numérica | Predictor (X) |
| `gaming_hours` | Horas diarias en videojuegos | Numérica | Predictor (X) |
| `work_study_hours` | Horas diarias de trabajo o estudio en pantalla | Numérica | Predictor (X) |
| `sleep_hours` | Horas de sueño por noche | Numérica | Predictor (X) |
| `notifications_per_day` | Notificaciones recibidas por día | Numérica | Predictor (X) |
| `app_opens_per_day` | Aperturas de aplicaciones por día | Numérica | Predictor (X) |
| `age` | Edad en años (18 a 35) | Numérica | Predictor (X) |
| `gender` | Género | Categórica (3 niveles) | Predictor (X) |
| `stress_level` | Nivel de estrés | Categórica (3 niveles) | Predictor (X) |
| `academic_work_impact` | Reporta impacto académico o laboral | Categórica (Sí / No) | Predictor (X) |
| `addicted_label` | Presenta uso problemático de pantallas | Binaria | Objetivo (Y) |
| `id` | Identificador de la persona | Identificador | No se usa |

### Preparación de los datos

El dataset venía limpio salvo por los valores faltantes, por lo que la preparación fue acotada:

- No se eliminan filas: los faltantes se conservan y se imputan dentro del pipeline de modelado del Avance 3, con valores aprendidos solo del conjunto de entrenamiento.
- Se excluye la columna `id`.
- Se agregan tres variables derivadas: `otros_usos_hours` (pantalla diaria menos redes, videojuegos y trabajo o estudio), `prop_redes` (proporción del tiempo de pantalla dedicado a redes sociales) y `dif_finde_hours` (pantalla de fin de semana menos pantalla de día hábil).

El resultado se guarda en `data/processed/adiccion_pantalla_procesado.csv.gz`. Es un CSV comprimido que pandas lee directamente con `pd.read_csv`.

## Estado del proyecto

| Avance | Contenido | Estado |
| :--- | :--- | :--- |
| 1 | Problema, pregunta, variables X, Y, T y dataset (`reports/adiccion_pantalla.pptx`) | Entregado |
| 2 | Análisis exploratorio de datos (`notebooks/02_eda.ipynb`) y aplicación Streamlit (`app/`) | Entregado |
| 3 | Modelado y evaluación (`notebooks/modelo.ipynb`), explicabilidad y conclusión (`notebooks/03_explicabilidad.ipynb`, `reports/avance3_explicabilidad_conclusion.pptx`) | Actual |

### Principales hallazgos del Avance 2

1. **La variable objetivo está desbalanceada:** 7 de cada 10 personas están etiquetadas con uso problemático.
2. **El tiempo de pantalla y las redes sociales concentran la información sobre Y:** la mediana de pantalla diaria es 9,1 h en quienes tienen uso problemático y 5,0 h en quienes no.
3. **La relación no es lineal:** sigue un efecto umbral y, bajo ese umbral, las redes sociales marcan la diferencia.
4. **El contexto personal y el sueño casi no se relacionan con Y**, un resultado que no esperábamos según la pregunta inicial.
5. **El contexto semanal (T) no distingue grupos:** todos aumentan su uso el fin de semana en la misma magnitud.

### Modelado del Avance 3

Se compararon tres modelos de complejidad creciente con el mismo esquema de evaluación:

- **Entradas:** los 12 predictores originales (las variables derivadas del Avance 2 no se usan en el modelo).
- **Partición:** 80 % entrenamiento (553.095 personas) y 20 % prueba (138.274), estratificada por Y y con `random_state=42`. El conjunto de prueba solo se usa para la evaluación final.
- **Preprocesamiento dentro del pipeline:** mediana para las numéricas, moda para las categóricas, codificación ordinal del nivel de estrés y *one-hot* para género e impacto académico o laboral. Todo se ajusta solo con entrenamiento.
- **Selección de hiperparámetros:** validación cruzada estratificada de 5 folds sobre una submuestra de 100.000 personas de entrenamiento, con la precisión promedio (AP) como criterio. Luego cada modelo se reentrena con todo el conjunto de entrenamiento.

| Modelo | Descripción | Configuración elegida |
| :--- | :--- | :--- |
| Regla simple | Árbol de decisión de profundidad 2 sobre pantalla diaria y redes sociales. Sirve como línea base | Con más de 6,5 h de pantalla predice uso problemático; bajo ese umbral, solo si las redes superan 1,5 h |
| Regresión logística | Splines sobre las numéricas más la interacción pantalla × redes, para captar el umbral y la interacción del EDA | `C = 0,1` y 6 nodos (`GridSearchCV`) |
| **LightGBM** | Ensamble de árboles con *boosting* | 500 árboles, tasa de aprendizaje 0,1, 31 hojas y mínimo 300 personas por hoja (`RandomizedSearchCV`, 15 combinaciones) |

**Resultados en el conjunto de prueba** (umbral 0,5; clase positiva = uso problemático, prevalencia 70,9 %):

| Métrica | Regla simple | Regresión logística | **LightGBM** |
| :--- | :---: | :---: | :---: |
| AP (precisión promedio) | 0,920 | 0,968 | **0,984** |
| ROC-AUC | 0,867 | 0,920 | **0,962** |
| Accuracy | 0,805 | 0,841 | **0,898** |
| Recall clase 1 (uso problemático) | 0,971 | 0,891 | 0,933 |
| Recall clase 0 (sin uso problemático) | 0,402 | 0,718 | **0,813** |
| AP entrenamiento / prueba | 0,920 / 0,920 | 0,969 / 0,968 | 0,987 / 0,984 |

Como referencia, predecir siempre "uso problemático" logra un accuracy de 0,709. Las métricas de validación cruzada sobre entrenamiento son prácticamente iguales a las de prueba.

El modelo recomendado se explicó con **SHAP** (`TreeExplainer`, salida en probabilidad): una vista global sobre 2.000 personas de prueba y tres explicaciones locales (un acierto, un falso negativo y un caso de contraste). Las figuras se guardan en `reports/figuras_avance3/`.

### Principales hallazgos del Avance 3

1. **LightGBM es el mejor modelo:** supera a los otros dos en todas las métricas resumen y es el único que reconoce bien a quienes *no* tienen uso problemático (recall de la clase 0 de 0,81, frente a 0,40 de la regla simple). La diferencia entre entrenamiento y prueba es mínima, sin señales de sobreajuste.
2. **El modelo decide según cuánto y en qué se usa la pantalla:** los hábitos de uso concentran el 77 % de la importancia SHAP (pantalla diaria 29 %, fin de semana 23 % y redes sociales 17 %). El contexto personal y el sueño suman solo el 4 %, lo que confirma el hallazgo 4 del Avance 2.
3. **El modelo reproduce el efecto umbral y la interacción del EDA:** la contribución de la pantalla diaria va de −21 puntos porcentuales bajo 4 h a +16 sobre 10 h y cambia de signo entre 6,5 y 7 h. Las redes sociales pesan más cuando la pantalla total es baja o media.
4. **Hallazgo nuevo:** a igual tiempo total de pantalla, más horas de trabajo o estudio, o más de 2 h de videojuegos, *bajan* la probabilidad. El riesgo está en el tiempo de ocio y redes, no en el tiempo productivo.
5. **Los errores se concentran en la zona intermedia:** LightGBM se equivoca en cerca del 16 % al 18 % de los casos bajo 8 h de pantalla y en un 0,1 % sobre 10 h. Ahí la decisión depende de varias variables a la vez: dos personas con casi el mismo tiempo de pantalla (5,3 h y 5,9 h) quedan en lados opuestos según sus horas de redes sociales.

### Conclusión

**Modelo recomendado: LightGBM.** Tiene el mejor rendimiento y sus explicaciones son coherentes con el EDA.

- **Limitación principal:** cerca del 19 % de su importancia viene de notificaciones y aperturas de apps. No tienen una dirección clara: sus valores exactos funcionan como una "huella" de perfiles, probablemente un artefacto de la generación sintética del dataset, que difícilmente se repetiría con datos reales.
- **Mejora propuesta:** entrenar LightGBM sin esas dos variables. En validación cruzada baja de AP 0,985 a 0,977 y sigue superando a la regresión logística con todas las variables (0,969). Después conviene validarlo con datos reales y ajustar el umbral según el costo de cada error, ya que un falso negativo deja a una persona sin intervención.
- **Límites de la interpretación:** SHAP explica al modelo y no implica causalidad; pantalla diaria y de fin de semana (r = 0,80) se reparten la importancia, y el 14,6 % de la muestra explicada tenía la pantalla diaria imputada.

**Respuesta a la pregunta del proyecto:** sí, es posible identificar el uso problemático de pantallas a partir de los hábitos digitales de una semana típica (ROC-AUC 0,96 en prueba). Lo que decide es cuánto se usa la pantalla y en qué (redes sociales), no el contexto personal ni el sueño.

## Estructura del repositorio

```
CD-2026-Proyecto-Equipo99/
├── app/                          Aplicación Streamlit
│   ├── Inicio.py                 Página de entrada
│   ├── utilidades.py             Carga de datos, colores y formatos compartidos
│   └── pages/
│       ├── 1_Problema_datos.py   ¿Qué estamos estudiando y con qué datos?
│       ├── 2_EDA.py              ¿Cómo se comportan nuestros datos?
│       └── 3_Hallazgos.py        ¿Qué hemos aprendido de nuestros datos?
├── data/
│   ├── raw/                      Datos originales (adiccion_pantalla.csv)
│   └── processed/                Datos preparados (adiccion_pantalla_procesado.csv.gz)
├── notebooks/
│   ├── 02_eda.ipynb              Análisis exploratorio completo
│   ├── modelo.ipynb              Comparación de modelos y evaluación (Avance 3)
│   └── 03_explicabilidad.ipynb   Explicabilidad con SHAP y conclusión (Avance 3)
├── reports/
│   ├── adiccion_pantalla.pptx    Presentación del Avance 1
│   ├── avance3_explicabilidad_conclusion.pptx   Presentación del Avance 3
│   └── figuras_avance3/          Gráficos SHAP globales y locales
├── src/                          Reservado para funciones reutilizables
├── requirements.txt              Dependencias del proyecto
└── README.md
```

## Cómo ejecutar el proyecto

Se necesita Python 3.10 o superior.

1. Clonar el repositorio y entrar a la carpeta:

   ```bash
   git clone https://github.com/Nervots2/CD-2026-Proyecto-Equipo99.git
   cd CD-2026-Proyecto-Equipo99
   ```

2. Crear y activar un entorno virtual (opcional, pero recomendado):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   ```

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Lanzar la aplicación desde la raíz del proyecto:

   ```bash
   streamlit run app/Inicio.py
   ```

   La aplicación se abre en el navegador (por defecto en `http://localhost:8501`) y las tres páginas aparecen en el menú lateral.

Para revisar el análisis completo, abrir `notebooks/02_eda.ipynb` con Jupyter o VS Code y ejecutarlo desde la carpeta `notebooks/`, ya que lee los datos con la ruta relativa `../data/raw/adiccion_pantalla.csv`.

Para revisar el Avance 3, ejecutar también desde `notebooks/` primero `modelo.ipynb` (comparación de modelos) y luego `03_explicabilidad.ipynb` (SHAP y conclusión). Ambos trabajan con los datos preparados de `data/processed/adiccion_pantalla_procesado.csv.gz`. `03_explicabilidad.ipynb` reconstruye por su cuenta el modelo recomendado con la misma partición y los mismos hiperparámetros, y guarda sus gráficos en `reports/figuras_avance3/`.

## Principales dependencias

| Librería | Uso |
| :--- | :--- |
| `pandas`, `numpy` | Carga y manipulación de los datos |
| `streamlit` | Aplicación interactiva |
| `plotly` | Gráficos de la aplicación |
| `matplotlib`, `seaborn` | Gráficos de los notebooks |
| `scipy` | Medidas de asociación del EDA |
| `scikit-learn` | Pipelines de preprocesamiento, validación cruzada, modelos base y métricas |
| `lightgbm` | Modelo recomendado |
| `shap` | Explicabilidad global y local del modelo |

Las versiones mínimas están en `requirements.txt`.
