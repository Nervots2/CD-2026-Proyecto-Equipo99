import streamlit as st

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

Bienvenido a la aplicación interactiva del Avance 2. Utiliza el **menú lateral izquierdo** para navegar por las siguientes secciones:

1. **Problema y Datos:** Resumen de nuestro Avance 1, contexto, definición de variables ($X, Y, T$) y decisiones de limpieza.
2. **Análisis Exploratorio:** Gráficos interactivos para explorar distribuciones y relaciones de las variables de nuestro dataset (`train.csv`).
3. **Hallazgos:** Conclusiones principales derivadas de nuestro análisis que guiarán el modelado del Avance 3.
""")