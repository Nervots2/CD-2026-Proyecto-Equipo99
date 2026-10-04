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

- No se eliminan filas: los faltantes se conservan y se imputarán dentro del pipeline de modelado del Avance 3.
- Se excluye la columna `id`.
- Se agregan tres variables derivadas: `otros_usos_hours` (pantalla diaria menos redes, videojuegos y trabajo o estudio), `prop_redes` (proporción del tiempo de pantalla dedicado a redes sociales) y `dif_finde_hours` (pantalla de fin de semana menos pantalla de día hábil).

El resultado se guarda en `data/processed/adiccion_pantalla_procesado.csv.gz`. Es un CSV comprimido que pandas lee directamente con `pd.read_csv`.

## Estado del proyecto

| Avance | Contenido | Estado |
| :--- | :--- | :--- |
| 1 | Problema, pregunta, variables X, Y, T y dataset (`reports/adiccion_pantalla.pptx`) | Entregado |
| 2 | Análisis exploratorio de datos (`notebooks/02_eda.ipynb`) y aplicación Streamlit (`app/`) | Actual |
| 3 | Modelado y evaluación | Pendiente |

### Principales hallazgos del Avance 2

1. **La variable objetivo está desbalanceada:** 7 de cada 10 personas están etiquetadas con uso problemático.
2. **El tiempo de pantalla y las redes sociales concentran la información sobre Y:** la mediana de pantalla diaria es 9,1 h en quienes tienen uso problemático y 5,0 h en quienes no.
3. **La relación no es lineal:** sigue un efecto umbral y, bajo ese umbral, las redes sociales marcan la diferencia.
4. **El contexto personal y el sueño casi no se relacionan con Y**, un resultado que no esperábamos según la pregunta inicial.
5. **El contexto semanal (T) no distingue grupos:** todos aumentan su uso el fin de semana en la misma magnitud.

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
│   └── 02_eda.ipynb              Análisis exploratorio completo
├── reports/
│   └── adiccion_pantalla.pptx    Presentación del Avance 1
├── src/                          Reservado para funciones reutilizables (Avance 3)
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

## Principales dependencias

| Librería | Uso |
| :--- | :--- |
| `pandas`, `numpy` | Carga y manipulación de los datos |
| `streamlit` | Aplicación interactiva |
| `plotly` | Gráficos de la aplicación |
| `matplotlib`, `seaborn` | Gráficos del notebook |
| `scipy`, `scikit-learn` | Medidas de asociación y chequeo preliminar del notebook |

Las versiones mínimas están en `requirements.txt`.
