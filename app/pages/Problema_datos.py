import streamlit as st

# Configuración básica de la página
st.set_page_config(page_title="Problema y Datos", page_icon="📚", layout="wide")

st.title("📚 Problema y Datos")
st.markdown("---")

# --- Título y Descripción ---
st.header("¿Qué estamos estudiando?")
st.markdown("""
**Título del proyecto:** Adicción a la Pantalla: Detección de uso problemático de pantallas en jóvenes adultos.

**Descripción breve del problema:** 
El uso intensivo de pantallas convive con menos horas de sueño y mayor estrés. Sin embargo, distinguir un uso alto de un uso problemático (adicción) sigue haciéndose "a ojo" porque no sabemos qué hábitos concretos marcan la diferencia. Nos enfocamos en personas de 18 a 35 años, donde una intervención de higiene digital todavía puede cambiar hábitos.

**Pregunta principal:** 
¿Es posible identificar si una persona de 18 a 35 años presenta un patrón de uso problemático de pantallas ($Y$), a partir de sus hábitos digitales, de sueño y de su contexto personal medidos en una semana típica ($X$), usando únicamente la información disponible al momento de la evaluación ($T$)?
""")

# --- Definición de Variables ---
st.header("Definición resumida de variables")
st.markdown("""
* **Predictores ($X$):** 9 variables numéricas de conducta digital y sueño (tiempo de pantalla, redes sociales, videojuegos, horas de sueño, notificaciones, etc.) y 3 categóricas de contexto (género, nivel de estrés, impacto académico o laboral).
* **Objetivo ($Y$):** `addicted_label`. Indicador binario (1 = sí presenta uso problemático, 0 = no presenta).
* **Contexto ($T$):** Evaluación transversal de una semana típica de uso. No representa una serie de tiempo ni proyección a futuro.
""")

st.markdown("---")

# --- Datos y Calidad ---
st.header("¿Con qué datos estamos trabajando?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estructura del Dataset")
    st.markdown("""
    * **Fuente:** Dataset público de Kaggle sobre hábitos de uso de pantalla. Para este análisis utilizamos el conjunto de entrenamiento (`train.csv`).
    * **Número de registros:** 691.369 observaciones.
    * **Número de variables:** 14 en total (1 ID, 12 predictores, 1 variable objetivo).
    * **Tipos principales:**
        * *Numéricas:* Edad, horas de pantalla (diario/fin de semana), horas de sueño, notificaciones, aperturas de apps.
        * *Categóricas:* Género, nivel de estrés, impacto académico/laboral.
        * *Binaria:* Etiqueta de adicción (0.0 y 1.0).
    """)
    
with col2:
    st.subheader("Calidad y Decisiones de Limpieza")
    st.markdown("""
    * **Calidad básica:** 
        * La variable objetivo ($Y$) está **completamente etiquetada** (0 nulos en `train.csv`).
        * Las variables predictoras ($X$) presentan entre un **4% y 19% de valores faltantes**.
        * No existen registros duplicados ni valores atípicos ilógicos (la edad va exactamente de 18 a 35 años, y las horas máximas diarias son consistentes).
    * **Principales decisiones de limpieza:** 
        * **No aplicar DropNA:** Se decidió no eliminar las filas con datos faltantes en los predictores para no reducir críticamente el tamaño de la muestra ni perder información valiosa de otros campos.
        * **Imputación futura:** El tratamiento de los nulos se delegará a un *pipeline* de imputación (mediana para numéricas, moda para categóricas) durante la fase de modelado en el Avance 3.
    """)