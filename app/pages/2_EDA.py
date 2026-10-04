import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utilidades import (
    CATEGORICAS, NUMERICAS, Y,
    aplicar_estilo, cargar_datos, colores, etiqueta, exigir_datos, fmt_dec, fmt_num, fmt_pct,
    nombre, resumen_general, tabla_datos,
)

# Configuración básica de la página
st.set_page_config(page_title="Análisis Exploratorio", page_icon="📊", layout="wide")

GRUPOS = {0: "Y = 0 (sin uso problemático)", 1: "Y = 1 (con uso problemático)"}

# Lectura de cada variable; {m0} y {m1} son las medianas (o proporciones) de cada grupo
LECTURAS = {
    "daily_screen_time_hours": "Distribución en campana centrada en valores altos: el uso intensivo es la norma. "
                               "Es la variable que más separa a los grupos (mediana de {m0} en Y = 0 frente a {m1} en Y = 1).",
    "weekend_screen_time": "Mismo patrón que la pantalla diaria, desplazado hacia valores más altos. "
                           "También separa con claridad a los grupos ({m0} frente a {m1}).",
    "social_media_hours": "Asimetría positiva: la mayoría tiene un uso moderado y existe una cola de uso intensivo. "
                          "El grupo Y = 1 duplica la mediana del grupo Y = 0 ({m1} frente a {m0}).",
    "gaming_hours": "Asimetría positiva, con una diferencia moderada entre grupos ({m0} frente a {m1}).",
    "work_study_hours": "Asimetría positiva, con una diferencia moderada entre grupos ({m0} frente a {m1}).",
    "sleep_hours": "Distribución casi uniforme entre 4,5 y 9 h, poco esperable en una población real. "
                   "Ambos grupos se superponen ({m0} frente a {m1}): el sueño no distingue el uso problemático.",
    "notifications_per_day": "Distribución casi uniforme en todo su rango. Ambos grupos se superponen "
                             "({m0} frente a {m1}).",
    "app_opens_per_day": "Distribución casi uniforme en todo su rango. Ambos grupos se superponen "
                         "({m0} frente a {m1}).",
    "age": "Todas las edades entre 18 y 35 años aparecen con frecuencia similar, y ambos grupos tienen "
           "la misma mediana ({m0} frente a {m1}).",
    "gender": "Tres categorías balanceadas (cerca de un tercio cada una), con la misma composición en ambos grupos.",
    "stress_level": "Tres niveles balanceados, con la misma composición en ambos grupos: el estrés no distingue "
                    "el uso problemático.",
    "academic_work_impact": "Dos categorías cercanas al 50 %, con la misma composición en ambos grupos.",
}


# --- Cálculos (cacheados para no recalcular en cada interacción) ---
@st.cache_data
def descriptivos():
    d = cargar_datos()[list(NUMERICAS)]
    return pd.DataFrame({
        "Variable": [NUMERICAS[c][0] for c in NUMERICAS],
        "Unidad": [NUMERICAS[c][1] for c in NUMERICAS],
        "Media": d.mean().to_numpy(),
        "Mediana": d.median().to_numpy(),
        "Desv. est.": d.std().to_numpy(),
        "Mín.": d.min().to_numpy(),
        "P25": d.quantile(0.25).to_numpy(),
        "P75": d.quantile(0.75).to_numpy(),
        "Máx.": d.max().to_numpy(),
        "Asimetría": d.skew().to_numpy(),
    })


@st.cache_data
def distribucion_numerica(col):
    """% de personas por tramo de la variable, en total y dentro de cada grupo de Y."""
    df = cargar_datos()
    ancho = NUMERICAS[col][2]
    d = df[[col, Y]].dropna()
    inicio = np.floor(d[col] / ancho) * ancho
    conteo = d.groupby([inicio, Y]).size().unstack(fill_value=0)
    # La edad es un número entero: el punto va sobre el valor y no al centro del tramo
    centros = conteo.index.to_numpy() + (0 if col == "age" else ancho / 2)
    return {
        "centros": centros,
        "pct_total": conteo.sum(axis=1).to_numpy() / len(d) * 100,
        "pct_0": conteo[0].to_numpy() / conteo[0].sum() * 100,
        "pct_1": conteo[1].to_numpy() / conteo[1].sum() * 100,
        "media": d[col].mean(),
        "mediana": d[col].median(),
        "mediana_0": d.loc[d[Y] == 0, col].median(),
        "mediana_1": d.loc[d[Y] == 1, col].median(),
        "pct_faltante": df[col].isna().mean(),
    }


@st.cache_data
def distribucion_categorica(col):
    """% de personas por categoría, en total y dentro de cada grupo de Y."""
    df = cargar_datos()
    traduccion = CATEGORICAS[col][1]
    d = df[[col, Y]].dropna()
    conteo = d.groupby([col, Y]).size().unstack(fill_value=0).reindex(list(traduccion))
    return {
        "categorias": [traduccion[c] for c in conteo.index],
        "pct_total": conteo.sum(axis=1).to_numpy() / len(d) * 100,
        "pct_0": conteo[0].to_numpy() / conteo[0].sum() * 100,
        "pct_1": conteo[1].to_numpy() / conteo[1].sum() * 100,
        "pct_faltante": df[col].isna().mean(),
    }


@st.cache_data
def contexto_semanal():
    d = cargar_datos()[["daily_screen_time_hours", "weekend_screen_time"]].dropna()
    dif = d["weekend_screen_time"] - d["daily_screen_time_hours"]
    return {
        "media_diaria": d["daily_screen_time_hours"].mean(),
        "media_finde": d["weekend_screen_time"].mean(),
        "aumento": dif.mean(),
        "pct_mas_finde": (dif > 0).mean(),
    }


st.title("📊 Análisis Exploratorio")
st.markdown("---")
exigir_datos()

res = resumen_general()
color = colores()

st.header("¿Cómo se comportan nuestros datos?")
st.markdown(
    "Esta página describe la variable objetivo, las variables predictoras y el contexto semanal. "
    "Las relaciones más importantes entre ellas se resumen en la página **Hallazgos**."
)

# --- Variable objetivo ---
st.subheader("Variable objetivo ($Y$): `addicted_label`")

n_y0 = res["n"] - res["n_y1"]
m1, m2, m3 = st.columns(3)
m1.metric("Con uso problemático (Y = 1)", fmt_pct(res["tasa_y"]), help=f"{fmt_num(res['n_y1'])} personas")
m2.metric("Sin uso problemático (Y = 0)", fmt_pct(1 - res["tasa_y"]), help=f"{fmt_num(n_y0)} personas")
m3.metric("Razón entre clases", f"{fmt_dec(res['n_y1'] / n_y0, 1)} : 1")

fig = go.Figure()
for clase, n_clase in ((1, res["n_y1"]), (0, n_y0)):
    fig.add_trace(go.Bar(
        x=[n_clase / res["n"] * 100], y=["Personas"], orientation="h", name=GRUPOS[clase],
        marker=dict(color=color[f"y{clase}"], line=dict(color=color["fondo"], width=2)),
        text=f"{fmt_pct(n_clase / res['n'])} · {fmt_num(n_clase)} personas", textposition="inside",
        insidetextanchor="middle", textfont=dict(color="#ffffff", size=14),
        hovertemplate=f"{GRUPOS[clase]}<br>{fmt_num(n_clase)} personas<extra></extra>",
    ))
aplicar_estilo(fig, "Distribución de la variable objetivo", "% de personas", "", alto=230,
               subtitulo="7 de cada 10 personas están etiquetadas con uso problemático")
fig.update_layout(barmode="stack", legend=dict(traceorder="normal"), margin=dict(r=40))
fig.update_xaxes(range=[0, 100.5], dtick=20, ticksuffix=" %")
fig.update_yaxes(showticklabels=False)
st.plotly_chart(fig)
tabla_datos(pd.DataFrame({
    "Grupo": [GRUPOS[1], GRUPOS[0]],
    "Personas": [fmt_num(res["n_y1"]), fmt_num(n_y0)],
    "% del total": [fmt_pct(res["tasa_y"]), fmt_pct(1 - res["tasa_y"])],
}))

st.markdown(f"""
**Interpretación:** la variable objetivo está **desbalanceada** y la clase mayoritaria es la problemática. No tiene valores faltantes ni valores extremos (solo toma 0 y 1). Un modelo que siempre prediga "uso problemático" acertaría el {fmt_pct(res["tasa_y"])} de las veces, por lo que la exactitud (*accuracy*) por sí sola sería engañosa en el Avance 3.
""")

st.markdown("---")

# --- Variables predictoras ---
st.subheader("Variables predictoras ($X$): estadísticas descriptivas")
st.dataframe(
    descriptivos().style.format(precision=2, decimal=",", thousands="."),
    hide_index=True,
)
st.markdown("""
**Interpretación por grupos de variables:**

* **Tiempo de pantalla (diario y fin de semana):** media y mediana casi iguales (asimetría cercana a 0) y valores altos; la mitad de las personas usa entre 5,5 y 9,8 h al día.
* **Redes sociales, videojuegos y trabajo o estudio:** asimetría positiva (≈ 0,5). La mayoría tiene un uso moderado y hay una cola de usuarios intensivos.
* **Edad, sueño, notificaciones y aperturas de apps:** se reparten de forma casi uniforme en todo su rango, algo poco esperable en una población real (indicio de datos sintéticos).
* **Escalas muy distintas** (horas frente a conteos de 15 a 250): los modelos sensibles a la escala necesitarán estandarización.
""")

# --- Explorador de variables (elemento interactivo) ---
st.subheader("Explora cada variable")

col_sel, col_modo = st.columns([2, 3])
with col_sel:
    var = st.selectbox("Variable:", list(NUMERICAS) + list(CATEGORICAS), format_func=nombre)
with col_modo:
    comparar = st.radio(
        "¿Qué quieres ver?", ["Todas las personas", "Comparar grupos de Y"], horizontal=True,
    ) == "Comparar grupos de Y"

es_numerica = var in NUMERICAS
dist = distribucion_numerica(var) if es_numerica else distribucion_categorica(var)
eje_x = dist["centros"] if es_numerica else dist["categorias"]

col_graf, col_lectura = st.columns([3, 2])
with col_graf:
    fig = go.Figure()
    if comparar:
        for clase in (0, 1):
            if es_numerica:
                fig.add_trace(go.Scatter(
                    x=eje_x, y=dist[f"pct_{clase}"], mode="lines", name=GRUPOS[clase],
                    line=dict(color=color[f"y{clase}"], width=2),
                    hovertemplate="%{y:.1f} %<extra>Y = " + str(clase) + "</extra>",
                ))
            else:
                fig.add_trace(go.Bar(
                    x=eje_x, y=dist[f"pct_{clase}"], name=GRUPOS[clase], width=0.25,
                    marker=dict(color=color[f"y{clase}"], cornerradius=4),
                    hovertemplate="%{x}: %{y:.1f} %<extra>Y = " + str(clase) + "</extra>",
                ))
        eje_y = "% de personas del grupo"
        fig.update_layout(hovermode="x unified", barmode="group", bargroupgap=0.08)
    else:
        fig.add_trace(go.Bar(
            x=eje_x, y=dist["pct_total"], name="Todas las personas",
            width=None if es_numerica else 0.35,
            marker=dict(color=color["y0"], cornerradius=4 if not es_numerica else 2),
            hovertemplate="%{x}: %{y:.1f} %<extra></extra>",
        ))
        eje_y = "% de personas"
        fig.update_layout(bargap=0.15)
    aplicar_estilo(fig, f"Distribución de {nombre(var).lower()}", etiqueta(var), eje_y, alto=400,
                   subtitulo="Por grupo de la variable objetivo" if comparar else "Todas las personas con dato",
                   leyenda=comparar)
    fig.update_yaxes(rangemode="tozero", ticksuffix=" %")
    st.plotly_chart(fig)

    datos = {"Tramo (centro)" if es_numerica else "Categoría": eje_x, "% del total": dist["pct_total"],
             "% en Y = 0": dist["pct_0"], "% en Y = 1": dist["pct_1"]}
    tabla_datos(pd.DataFrame(datos).style.format(precision=2, decimal=",", thousands="."))

with col_lectura:
    if es_numerica:
        unidad = NUMERICAS[var][1]
        c1, c2 = st.columns(2)
        c1.metric("Media", f"{fmt_dec(dist['media'])} {unidad}")
        c2.metric("Mediana", f"{fmt_dec(dist['mediana'])} {unidad}")
        c1.metric("Mediana en Y = 0", f"{fmt_dec(dist['mediana_0'])} {unidad}")
        c2.metric("Mediana en Y = 1", f"{fmt_dec(dist['mediana_1'])} {unidad}")
        m0 = f"{fmt_dec(dist['mediana_0'])} {unidad}"
        m1 = f"{fmt_dec(dist['mediana_1'])} {unidad}"
    else:
        m0 = m1 = ""
    st.metric("Valores faltantes", fmt_pct(dist["pct_faltante"]))
    st.markdown(f"**Lectura:** {LECTURAS[var].format(m0=m0, m1=m1)}")
    if es_numerica:
        ancho = NUMERICAS[var][2]
        st.caption(f"Tramos de {fmt_dec(ancho, 2 if ancho < 0.5 else 1 if ancho < 1 else 0)} {NUMERICAS[var][1]}. "
                   "Los porcentajes se calculan sobre las personas que tienen el dato.")
    else:
        st.caption("Los porcentajes se calculan sobre las personas que tienen el dato.")

st.markdown("---")

# --- Contexto temporal ---
ctx = contexto_semanal()
st.subheader("Contexto ($T$): día hábil frente a fin de semana")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Pantalla en día hábil (media)", f"{fmt_dec(ctx['media_diaria'])} h")
t2.metric("Pantalla en fin de semana (media)", f"{fmt_dec(ctx['media_finde'])} h")
t3.metric("Aumento en fin de semana", f"+{fmt_dec(ctx['aumento'])} h")
t4.metric("Usa más pantalla el fin de semana", fmt_pct(ctx["pct_mas_finde"]))
st.markdown(
    "**Interpretación:** el dataset no tiene fechas, por lo que el único contraste temporal disponible es entre "
    "día hábil y fin de semana dentro de una semana típica. El uso sube en promedio cerca de 2 horas el fin de "
    "semana. Si ese aumento es distinto en quienes tienen uso problemático se revisa en **Hallazgos** (H5)."
)
