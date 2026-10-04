import pandas as pd
import streamlit as st

from utilidades import (
    CATEGORICAS, COMPONENTES, NUMERICAS, PREGUNTA, Y,
    cargar_datos, exigir_datos, fmt_dec, fmt_num, fmt_pct, resumen_general,
)

# Configuración básica de la página
st.set_page_config(page_title="Problema y Datos", page_icon="📚", layout="wide")

# Descripción de cada variable del dataset: (descripción, tipo, rol)
VARIABLES = {
    "daily_screen_time_hours": ("Horas de pantalla en un día hábil", "Numérica", "Predictor (X)"),
    "weekend_screen_time": ("Horas de pantalla en un día de fin de semana", "Numérica", "Predictor (X) · contexto T"),
    "social_media_hours": ("Horas diarias en redes sociales", "Numérica", "Predictor (X)"),
    "gaming_hours": ("Horas diarias en videojuegos", "Numérica", "Predictor (X)"),
    "work_study_hours": ("Horas diarias de trabajo o estudio en pantalla", "Numérica", "Predictor (X)"),
    "sleep_hours": ("Horas de sueño por noche", "Numérica", "Predictor (X)"),
    "notifications_per_day": ("Notificaciones recibidas por día", "Numérica", "Predictor (X)"),
    "app_opens_per_day": ("Aperturas de aplicaciones por día", "Numérica", "Predictor (X)"),
    "age": ("Edad en años", "Numérica", "Predictor (X)"),
    "gender": ("Género", "Categórica (3 niveles)", "Predictor (X)"),
    "stress_level": ("Nivel de estrés", "Categórica (3 niveles)", "Predictor (X)"),
    "academic_work_impact": ("Reporta impacto académico o laboral", "Categórica (Sí / No)", "Predictor (X)"),
    Y: ("Presenta uso problemático de pantallas (1 = sí, 0 = no)", "Binaria", "Objetivo (Y)"),
    "id": ("Identificador de la persona", "Identificador", "No se usa"),
}


# --- Cálculos (cacheados para no recalcular en cada interacción) ---
@st.cache_data
def tabla_variables():
    df = cargar_datos()
    return pd.DataFrame({
        "Variable": list(VARIABLES),
        "Descripción": [v[0] for v in VARIABLES.values()],
        "Tipo": [v[1] for v in VARIABLES.values()],
        "Rol": [v[2] for v in VARIABLES.values()],
        "% faltantes": [df[col].isna().mean() * 100 for col in VARIABLES],
    })


@st.cache_data
def calidad():
    df = cargar_datos()
    predictores = list(NUMERICAS) + list(CATEGORICAS)
    faltantes = df[predictores].isna().mean()

    # Valores atípicos con la regla del rango intercuartílico (IQR)
    pct_atipicos = {}
    for col in NUMERICAS:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        pct_atipicos[col] = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()

    # Coherencia interna: el tiempo total debe ser mayor o igual que la suma de sus componentes
    comp = df[["daily_screen_time_hours"] + COMPONENTES].dropna()
    coherente = (comp["daily_screen_time_hours"] >= comp[COMPONENTES].sum(axis=1) - 0.01).mean()

    # ¿La tasa de Y cambia cuando falta un dato?
    dif_nulos = max(abs(df.loc[df[c].isna(), Y].mean() - df.loc[df[c].notna(), Y].mean()) for c in predictores)

    return {
        "falt_min": faltantes.min(),
        "falt_max": faltantes.max(),
        "duplicados": int(df.drop(columns="id").duplicated().sum()),
        "atipicos_max": max(pct_atipicos.values()),
        "edad_min": df["age"].min(),
        "edad_max": df["age"].max(),
        "pct_coherente": coherente,
        "dif_nulos": dif_nulos,
    }


st.title("📚 Problema y Datos")
st.markdown("---")
exigir_datos()

res = resumen_general()
cal = calidad()

# --- Título y Descripción ---
st.header("¿Qué estamos estudiando?")
st.markdown("""
**Título del proyecto:** Adicción a la Pantalla: Detección de uso problemático de pantallas en jóvenes adultos.

**Descripción breve del problema:**
El uso intensivo de pantallas suele asociarse con menos horas de sueño y mayor estrés. Sin embargo, distinguir un uso alto de un uso problemático (adicción) sigue haciéndose "a ojo" porque no sabemos qué hábitos concretos marcan la diferencia. Nos enfocamos en personas de 18 a 35 años, donde una intervención de higiene digital todavía puede cambiar hábitos.
""")
st.info(f"**Pregunta principal:** {PREGUNTA}")

# --- Definición de Variables ---
st.header("Definición resumida de variables")
col_x, col_y, col_t = st.columns(3)
with col_x:
    st.markdown("""
**Predictores ($X$)**

9 variables numéricas de conducta digital y sueño (tiempo de pantalla, redes sociales, videojuegos, trabajo o estudio, horas de sueño, notificaciones, aperturas de apps y edad) y 3 categóricas de contexto (género, nivel de estrés, impacto académico o laboral).
""")
with col_y:
    st.markdown("""
**Objetivo ($Y$)**

`addicted_label`: indicador binario de uso problemático de pantallas (1 = sí presenta, 0 = no presenta). Es un problema de clasificación.
""")
with col_t:
    st.markdown("""
**Contexto ($T$)**

Una semana típica de uso, con contraste entre días hábiles y fin de semana. La evaluación es transversal: todas las variables describen el estado actual de la persona, sin serie de tiempo ni proyección a futuro.
""")

st.markdown("---")

# --- Datos ---
st.header("¿Con qué datos estamos trabajando?")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Registros", fmt_num(res["n"]), help="Una persona por fila.")
m2.metric("Variables", res["n_vars"], help="1 identificador, 12 predictores y 1 variable objetivo.")
m3.metric("Con uso problemático (Y = 1)", fmt_pct(res["tasa_y"]))
m4.metric("Filas sin ningún valor faltante", fmt_pct(res["filas_completas"]))

st.markdown(f"""
* **Fuente:** dataset público de una competición de Kaggle sobre hábitos de uso de pantalla (`data/raw/adiccion_pantalla.csv`).
* **Unidad de observación:** una persona por fila, de entre {fmt_dec(cal["edad_min"], 0)} y {fmt_dec(cal["edad_max"], 0)} años.
* **Tipos principales:** 9 variables numéricas, 3 categóricas y 1 binaria (la variable objetivo).
* **Cambio respecto del Avance 1:** la competición incluía además 296.302 registros sin etiqueta. Se descartaron porque no permiten evaluar un modelo, por lo que trabajamos con los {fmt_num(res["n"])} registros etiquetados.
""")

st.subheader("Variables del dataset")
st.dataframe(
    tabla_variables().style.format({"% faltantes": "{:.1f} %"}, decimal=","),
    hide_index=True,
    height=35 * (len(VARIABLES) + 1) + 3,  # alto suficiente para mostrar todas las filas sin desplazarse
)

st.markdown("---")

# --- Calidad y decisiones ---
st.header("Calidad de los datos y decisiones de limpieza")
st.markdown(
    "El dataset venía limpio salvo por los valores faltantes, así que la preparación fue acotada. "
    "Esto es lo que revisamos y lo que decidimos en cada caso:"
)

decisiones = pd.DataFrame({
    "Revisión": [
        "Valores faltantes",
        "¿Los faltantes dependen de Y?",
        "Registros duplicados",
        "Valores fuera de rango",
        "Valores atípicos",
        "Categorías inconsistentes",
        "Coherencia entre variables",
        "Columna id",
    ],
    "Qué encontramos": [
        f"Entre {fmt_pct(cal['falt_min'])} y {fmt_pct(cal['falt_max'])} por predictor; solo el "
        f"{fmt_pct(res['filas_completas'])} de las filas está completa. Y no tiene faltantes.",
        f"No: la tasa de Y cambia a lo más {fmt_dec(cal['dif_nulos'] * 100)} p.p. cuando falta un dato.",
        f"{cal['duplicados']} registros duplicados.",
        "Ninguno: edad entre 18 y 35 años y horas dentro de límites posibles.",
        f"A lo más el {fmt_pct(cal['atipicos_max'], 2)} de los valores de una variable (regla IQR), todos plausibles.",
        "Ninguna: 2 o 3 niveles por variable, sin variantes de escritura.",
        f"En el {fmt_pct(cal['pct_coherente'], 0)} de los casos la pantalla diaria es mayor o igual que la suma "
        "de redes, videojuegos y trabajo/estudio.",
        "Identificador único por fila, sin información.",
    ],
    "Decisión": [
        "No eliminar filas. Se conservan los faltantes y se imputarán dentro del pipeline del Avance 3.",
        "Una imputación simple (mediana / moda) no debería sesgar la relación con Y.",
        "Nada que eliminar.",
        "Nada que corregir.",
        "Se mantienen, sin recorte ni eliminación.",
        "Nada que corregir.",
        "Se crean variables derivadas (otros usos, proporción en redes, diferencia fin de semana − diario).",
        "Se excluye del análisis y del modelado.",
    ],
})
# Se muestra como tabla de texto para que las celdas largas no se corten
filas = ["| " + " | ".join(decisiones.columns) + " |", "|---|---|---|"]
filas += ["| **" + f[0] + "** | " + f[1] + " | " + f[2] + " |" for f in decisiones.itertuples(index=False)]
st.markdown("\n".join(filas))

st.caption(
    "Un valor faltante no significa cero: hay personas sin dato de pantalla diaria que sí registran horas en redes "
    "sociales o videojuegos. El dataset preparado (sin `id` y con las variables derivadas) se guarda en "
    "`data/processed/adiccion_pantalla_procesado.csv.gz`."
)
