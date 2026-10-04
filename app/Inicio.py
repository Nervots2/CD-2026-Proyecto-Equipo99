import streamlit as st

from utilidades import PREGUNTA

st.set_page_config(
    page_title="Proyecto CD - Avance 2",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Proyecto: Adicción a la Pantalla en Jóvenes Adultos")
st.subheader("Avance 2: Análisis Exploratorio de Datos (EDA)")

st.markdown("""
**Universidad Técnica Federico Santa María**
**Integrantes:** Javier Andrade, Lucas Martínez, Benjamín Ramos
""")

st.info(f"**Pregunta del proyecto:** {PREGUNTA}")

st.markdown("""
Bienvenido a la aplicación interactiva del Avance 2. Utiliza el **menú lateral izquierdo** para navegar por las siguientes secciones:

1. **Problema y Datos:** Resumen de nuestro Avance 1, contexto, definición de variables ($X, Y, T$), calidad de los datos y decisiones de limpieza.
2. **Análisis Exploratorio (EDA):** Comportamiento de la variable objetivo, estadísticas descriptivas y un explorador interactivo de cada variable del dataset.
3. **Hallazgos:** Los cinco hallazgos principales del análisis, con su evidencia, las limitaciones de los datos y lo que implican para el modelado del Avance 3.
""")
