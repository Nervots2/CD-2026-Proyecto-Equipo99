from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configuración básica de la página
st.set_page_config(page_title="Hallazgos", page_icon="🔎", layout="wide")

# --- Constantes ---
Y = "addicted_label"

# Rutas posibles del dataset (raíz del proyecto o estructura data/raw del repositorio)
RAIZ = Path(__file__).resolve().parents[1]
RUTAS_DATOS = [
    RAIZ / "adiccion_pantalla.csv",
    RAIZ / "data" / "raw" / "adiccion_pantalla.csv",
    RAIZ.parent / "data" / "raw" / "adiccion_pantalla.csv",
]

# Colores: el naranjo representa siempre a Y = 1 (uso problemático) en toda la página
AZUL = "#2a78d6"
NARANJO = "#eb6834"
AZUL_CLARO = "#86b6ef"
GRIS = "#c3c2b7"
RAMPA_ORDINAL = ["#86b6ef", "#3987e5", "#184f95"]
ESCALA_SECUENCIAL = [
    [0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"],
]

# Variables numéricas: (nombre legible, unidad, ancho del tramo para agrupar)
NUMERICAS = {
    "daily_screen_time_hours": ("Tiempo de pantalla diario", "h/día", 0.5),
    "weekend_screen_time": ("Tiempo de pantalla en fin de semana", "h/día", 0.5),
    "social_media_hours": ("Redes sociales", "h/día", 0.5),
    "gaming_hours": ("Videojuegos", "h/día", 0.5),
    "work_study_hours": ("Trabajo o estudio en pantalla", "h/día", 0.5),
    "sleep_hours": ("Horas de sueño", "h/noche", 0.5),
    "notifications_per_day": ("Notificaciones", "por día", 10),
    "app_opens_per_day": ("Aperturas de apps", "por día", 10),
    "age": ("Edad", "años", 1),
}

# Variables categóricas: (nombre legible, traducción y orden de las categorías)
CATEGORICAS = {
    "gender": ("Género", {"Male": "Hombre", "Female": "Mujer", "Other": "Otro"}),
    "stress_level": ("Nivel de estrés", {"Low": "Bajo", "Medium": "Medio", "High": "Alto"}),
    "academic_work_impact": ("Impacto académico/laboral", {"No": "No", "Yes": "Sí"}),
}

MIN_OBS_TRAMO = 200  # tramos con menos observaciones se omiten para no graficar ruido


# --- Formato de números (estilo chileno: 691.369 y 70,9 %) ---
def fmt_num(x):
    return f"{x:,.0f}".replace(",", ".")


def fmt_dec(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


def fmt_pct(x, dec=1):
    return f"{x * 100:.{dec}f}".replace(".", ",") + " %"


# --- Carga de datos y cálculos (cacheados para no recalcular en cada interacción) ---
@st.cache_data
def cargar_datos():
    for ruta in RUTAS_DATOS:
        if ruta.exists():
            return pd.read_csv(ruta)
    return None


def auc(score, y):
    """AUC de Mann-Whitney: probabilidad de que un caso Y=1 tenga mayor score que uno Y=0."""
    rangos = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return (rangos[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cramer_v(x, y):
    tabla = pd.crosstab(x, y).to_numpy()
    esperado = tabla.sum(axis=1, keepdims=True) * tabla.sum(axis=0, keepdims=True) / tabla.sum()
    chi2 = ((tabla - esperado) ** 2 / esperado).sum()
    return np.sqrt(chi2 / (tabla.sum() * (min(tabla.shape) - 1)))


@st.cache_data
def resumen_general():
    df = cargar_datos()
    return {
        "n": len(df),
        "tasa_y": df[Y].mean(),
        "n_y1": int(df[Y].sum()),
        "filas_completas": df.dropna().shape[0] / len(df),
    }


@st.cache_data
def ranking_predictores():
    """AUC univariado de cada predictor: 0,5 = no separa mejor que el azar, 1 = separación perfecta."""
    df = cargar_datos()
    filas = []
    for col, (nombre, _, _) in NUMERICAS.items():
        d = df[[col, Y]].dropna()
        a = auc(d[col].to_numpy(), d[Y].to_numpy())
        filas.append({"variable": col, "nombre": nombre, "tipo": "Numérica", "auc": max(a, 1 - a), "cramer_v": np.nan})
    for col, (nombre, _) in CATEGORICAS.items():
        d = df[[col, Y]].dropna()
        # Se usa la tasa de Y de cada categoría como score para obtener un AUC comparable
        score = d[col].map(d.groupby(col)[Y].mean())
        a = auc(score.to_numpy(), d[Y].to_numpy())
        filas.append({"variable": col, "nombre": nombre, "tipo": "Categórica", "auc": max(a, 1 - a), "cramer_v": cramer_v(d[col], d[Y])})
    return pd.DataFrame(filas).sort_values("auc").reset_index(drop=True)


@st.cache_data
def distribucion_por_clase(col):
    df = cargar_datos()
    ancho = NUMERICAS[col][2]
    d = df[[col, Y]].dropna()
    bordes = np.arange(np.floor(d[col].min() / ancho) * ancho, d[col].max() + ancho, ancho)
    centros = (bordes[:-1] + bordes[1:]) / 2
    salida = {"centros": centros}
    for clase in (0, 1):
        valores = d.loc[d[Y] == clase, col]
        conteo, _ = np.histogram(valores, bins=bordes)
        salida[f"pct_{clase}"] = conteo / conteo.sum() * 100
        salida[f"mediana_{clase}"] = valores.median()
    return salida


@st.cache_data
def tasa_por_tramo(col):
    df = cargar_datos()
    ancho = NUMERICAS[col][2]
    d = df[[col, Y]].dropna()
    tramo = np.floor(d[col] / ancho) * ancho
    t = d.groupby(tramo)[Y].agg(tasa="mean", n="size").reset_index(names="inicio")
    t["centro"] = t["inicio"] + ancho / 2
    return t[t["n"] >= MIN_OBS_TRAMO]


TRAMOS_PANTALLA = ([0, 4, 6, 8, 10, 24], ["< 4 h", "4–6 h", "6–8 h", "8–10 h", "≥ 10 h"])
TRAMOS_REDES = ([0, 1, 2, 3, 4, 24], ["< 1 h", "1–2 h", "2–3 h", "3–4 h", "≥ 4 h"])


@st.cache_data
def mapa_pantalla_redes():
    df = cargar_datos()
    d = df[["daily_screen_time_hours", "social_media_hours", Y]].dropna()
    pantalla = pd.cut(d["daily_screen_time_hours"], TRAMOS_PANTALLA[0], labels=TRAMOS_PANTALLA[1], right=False)
    redes = pd.cut(d["social_media_hours"], TRAMOS_REDES[0], labels=TRAMOS_REDES[1], right=False)
    agrupado = d.groupby([pantalla, redes], observed=False)[Y]
    tasa = agrupado.mean().unstack()
    n = agrupado.size().unstack()
    return tasa.where(n >= 100), n


@st.cache_data
def tasa_por_categoria(col):
    df = cargar_datos()
    nombres = CATEGORICAS[col][1]
    d = df[[col, Y]].dropna()
    t = d.groupby(col)[Y].agg(tasa="mean", n="size").reindex(list(nombres))
    t.index = [nombres[c] for c in t.index]
    return t


@st.cache_data
def umbral_por_estres():
    df = cargar_datos()
    d = df[["daily_screen_time_hours", "stress_level", Y]].dropna()
    tramo = np.floor(d["daily_screen_time_hours"])
    t = d.groupby([tramo, "stress_level"])[Y].agg(tasa="mean", n="size").reset_index()
    t = t[t["n"] >= MIN_OBS_TRAMO]
    t["centro"] = t["daily_screen_time_hours"] + 0.5
    return t


@st.cache_data
def semana_vs_finde():
    df = cargar_datos()
    d = df[["daily_screen_time_hours", "weekend_screen_time", Y]].dropna()
    dif = d["weekend_screen_time"] - d["daily_screen_time_hours"]
    bordes = np.arange(-6, 8.5, 0.5)
    salida = {
        "centros": (bordes[:-1] + bordes[1:]) / 2,
        "corr": d["daily_screen_time_hours"].corr(d["weekend_screen_time"]),
        "dif_media": dif.mean(),
    }
    for clase in (0, 1):
        valores = dif[d[Y] == clase]
        conteo, _ = np.histogram(valores, bins=bordes)
        salida[f"pct_{clase}"] = conteo / len(valores) * 100
        salida[f"media_{clase}"] = valores.mean()
    bx = np.arange(0, 15.5, 0.5)
    by = np.arange(0, 18.5, 0.5)
    conteo2d, _, _ = np.histogram2d(d["daily_screen_time_hours"], d["weekend_screen_time"], bins=[bx, by])
    salida["mapa"] = conteo2d.T
    salida["bx"] = (bx[:-1] + bx[1:]) / 2
    salida["by"] = (by[:-1] + by[1:]) / 2
    return salida


@st.cache_data
def calidad_y_limitaciones():
    df = cargar_datos()
    predictores = list(NUMERICAS) + list(CATEGORICAS)
    tasas_nulos = [df.loc[df[c].isna(), Y].mean() for c in predictores]
    tasas_obs = [df.loc[df[c].notna(), Y].mean() for c in predictores]
    comp = df[["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours"]].dropna()
    suma = comp[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(axis=1)
    top_aperturas = df["app_opens_per_day"].value_counts()
    return {
        "dif_max_nulos": max(abs(a - b) for a, b in zip(tasas_nulos, tasas_obs)),
        "tasa_nulos_min": min(tasas_nulos),
        "tasa_nulos_max": max(tasas_nulos),
        "pct_suma_coherente": (comp["daily_screen_time_hours"] >= suma - 0.01).mean(),
        "corr_suma": suma.corr(comp["daily_screen_time_hours"]),
        "valor_aperturas_top": top_aperturas.index[0],
        "n_aperturas_top": int(top_aperturas.iloc[0]),
    }


# --- Utilidades de presentación ---
def tema_oscuro():
    try:
        return st.context.theme.type == "dark"
    except AttributeError:
        return False


def escala_secuencial():
    # En modo oscuro se invierte la rampa para que los valores bajos se confundan con el fondo
    if tema_oscuro():
        return [[pos, color] for pos, (_, color) in zip([p for p, _ in ESCALA_SECUENCIAL], reversed(ESCALA_SECUENCIAL))]
    return ESCALA_SECUENCIAL


def color_texto_celda(proporcion):
    """Texto oscuro sobre celdas claras y blanco sobre celdas oscuras, según el tema activo."""
    celda_oscura = (proporcion > 0.5) != tema_oscuro()
    return "#ffffff" if celda_oscura else "#0b0b0b"


def aplicar_estilo(fig, titulo, eje_x, eje_y, alto=420, subtitulo=None):
    fig.update_layout(
        # Título fijo arriba del contenedor para dejar espacio a la leyenda (que puede ocupar 2 filas)
        title=dict(text=titulo, font=dict(size=16), subtitle=dict(text=subtitulo or ""),
                   yref="container", y=1, yanchor="top", pad=dict(t=10)),
        xaxis_title=eje_x,
        yaxis_title=eje_y,
        height=alto,
        separators=",.",
        margin=dict(t=130, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


def linea_promedio(fig, tasa):
    fig.add_hline(
        y=tasa * 100, line=dict(color="#898781", width=1, dash="dot"),
        annotation_text=f"Promedio global: {fmt_pct(tasa)}", annotation_position="top left",
        annotation_font=dict(color="#898781", size=11),
    )


def bloque_hallazgo(evidencia, patron, interpretacion, modelado):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**📊 Evidencia**\n\n{evidencia}")
    with c2:
        st.markdown(f"**🔁 Patrón**\n\n{patron}")
    with c3:
        st.markdown(f"**💡 Interpretación**\n\n{interpretacion}")
    with c4:
        st.markdown(f"**🎯 Para el modelado**\n\n{modelado}")


# --- Encabezado ---
st.title("🔎 Hallazgos")
st.markdown("---")

if cargar_datos() is None:
    st.error(
        "No se encontró `adiccion_pantalla.csv`. Colócalo en la raíz del proyecto o en `data/raw/`."
    )
    st.stop()

res = resumen_general()
rank = ranking_predictores().set_index("variable")
mapa_tasa, mapa_n = mapa_pantalla_redes()
finde = semana_vs_finde()
calidad = calidad_y_limitaciones()
curva_pantalla = tasa_por_tramo("daily_screen_time_hours")

# Valores clave del efecto umbral, calculados desde los datos
inicio_subida = curva_pantalla.loc[curva_pantalla["tasa"] >= 0.4, "inicio"].min()
inicio_umbral = curva_pantalla.loc[curva_pantalla["tasa"] >= 0.5, "inicio"].min()
fin_umbral = curva_pantalla.loc[curva_pantalla["tasa"] >= 0.95, "inicio"].min()
tasa_bajo_4h = curva_pantalla.loc[curva_pantalla["inicio"] < 4, "tasa"]
tasas_estres = tasa_por_categoria("stress_level")["tasa"]
tasas_genero = tasa_por_categoria("gender")["tasa"]
v_max_categoricas = rank.loc[list(CATEGORICAS), "cramer_v"].max()
auc_debiles = rank.loc[["age", "sleep_hours", "notifications_per_day", "app_opens_per_day"], "auc"]

st.header("¿Qué hemos aprendido de nuestros datos?")
st.markdown(
    "Esta página resume los resultados más importantes del EDA. Cada hallazgo sigue la secuencia "
    "**Evidencia → Patrón → Interpretación → relación con la pregunta**, y termina con lo que implica "
    "para el modelado del Avance 3."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Registros analizados", fmt_num(res["n"]))
m2.metric("Con uso problemático (Y = 1)", fmt_pct(res["tasa_y"]))
m3.metric("Mejor predictor (AUC)", fmt_dec(rank.loc["daily_screen_time_hours", "auc"]),
          help="Tiempo de pantalla diario. AUC = 0,5 equivale al azar; 1 es separación perfecta.")
m4.metric("Filas sin ningún valor faltante", fmt_pct(res["filas_completas"]))

# --- Resumen explícito de los hallazgos ---
st.subheader("Resumen de hallazgos")
st.markdown(f"""
1. **La variable objetivo está desbalanceada:** {fmt_pct(res["tasa_y"])} de las personas presenta uso problemático (≈ 7 de cada 10).
2. **El tiempo de pantalla y las redes sociales son las variables que más informan sobre Y**, con mucha diferencia sobre el resto.
3. **La relación entre tiempo de pantalla e Y no es lineal:** sigue un efecto umbral (curva en S) y, bajo ese umbral, las redes sociales marcan la diferencia.
4. **Las variables de contexto personal y de sueño casi no se relacionan con Y**, un resultado que no esperábamos según nuestra pregunta inicial.
5. **El contexto semanal (T) no distingue grupos:** todos aumentan su uso en el fin de semana en la misma magnitud, por lo que ambas medidas de pantalla son redundantes.
""")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "H1 · Y desbalanceada",
    "H2 · Predictores clave",
    "H3 · Efecto umbral",
    "H4 · Contexto sin señal",
    "H5 · Semana vs fin de semana",
])

# --- Hallazgo 1: desbalance de Y ---
with tab1:
    st.subheader("Hallazgo 1: 7 de cada 10 personas están etiquetadas con uso problemático")

    n_y0 = res["n"] - res["n_y1"]
    fig = go.Figure(go.Bar(
        x=[n_y0, res["n_y1"]],
        y=["Sin uso problemático (Y = 0)", "Con uso problemático (Y = 1)"],
        orientation="h",
        marker=dict(color=[AZUL, NARANJO], cornerradius=4),
        text=[f"{fmt_num(n_y0)} ({fmt_pct(1 - res['tasa_y'])})", f"{fmt_num(res['n_y1'])} ({fmt_pct(res['tasa_y'])})"],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:,.0f} personas<extra></extra>",
        width=0.55,
    ))
    aplicar_estilo(fig, "Distribución de la variable objetivo", "Número de personas", "", alto=320,
                   subtitulo="addicted_label · 1 = uso problemático, 0 = sin uso problemático")
    fig.update_xaxes(range=[0, res["n_y1"] * 1.3])
    st.plotly_chart(fig)

    bloque_hallazgo(
        evidencia=f"{fmt_num(res['n_y1'])} personas con Y = 1 ({fmt_pct(res['tasa_y'])}) frente a "
                  f"{fmt_num(n_y0)} con Y = 0. La etiqueta no tiene valores faltantes.",
        patron="La clase *con uso problemático* es la mayoritaria, en proporción ≈ 2,4 : 1.",
        interpretacion=f"Un modelo que siempre prediga \"adicto\" ya acertaría el {fmt_pct(res['tasa_y'])} de las veces, "
        "por lo que el *accuracy* por sí solo sería engañoso.",
        modelado="Usar partición estratificada, métricas como F1, recall de la clase 0, ROC-AUC y PR-AUC, "
                 "y evaluar `class_weight` o ajuste del umbral de decisión.",
    )

# --- Hallazgo 2: predictores clave ---
with tab2:
    st.subheader("Hallazgo 2: el tiempo de pantalla y las redes sociales concentran la información sobre Y")

    col_izq, col_der = st.columns(2)
    with col_izq:
        colores = [AZUL if a >= 0.8 else AZUL_CLARO if a >= 0.6 else GRIS for a in rank["auc"]]
        # Las barras parten en 0,5 (azar), así su largo representa la capacidad de separar Y
        fig = go.Figure(go.Bar(
            x=rank["auc"] - 0.5, base=0.5, y=rank["nombre"], orientation="h",
            marker=dict(color=colores, cornerradius=4),
            customdata=np.stack([rank["tipo"], rank["auc"]], axis=-1),
            hovertemplate="%{y} (%{customdata[0]})<br>AUC = %{customdata[1]:.3f}<extra></extra>",
        ))
        aplicar_estilo(fig, "AUC univariado por predictor", "AUC univariado (0,5 = azar)", "", alto=460,
                       subtitulo="Capacidad individual de cada variable para separar Y")
        fig.update_xaxes(range=[0.5, 0.95], dtick=0.1)
        st.plotly_chart(fig)
        st.caption("Azul oscuro: AUC ≥ 0,8 (fuerte) · azul claro: 0,6–0,8 (moderado) · gris: < 0,6 (débil). "
                   "Para las categóricas se usó la tasa de Y de cada categoría como puntaje.")

    with col_der:
        opciones = list(NUMERICAS)
        var = st.selectbox(
            "Compara la distribución de una variable entre ambos grupos:",
            opciones, index=0, format_func=lambda c: NUMERICAS[c][0], key="sel_h2",
        )
        nombre, unidad, _ = NUMERICAS[var]
        dist = distribucion_por_clase(var)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dist["centros"], y=dist["pct_0"], mode="lines", name="Y = 0 (sin uso problemático)",
                                 line=dict(color=AZUL, width=2), hovertemplate="%{x}: %{y:.1f} %<extra>Y = 0</extra>"))
        fig.add_trace(go.Scatter(x=dist["centros"], y=dist["pct_1"], mode="lines", name="Y = 1 (con uso problemático)",
                                 line=dict(color=NARANJO, width=2), hovertemplate="%{x}: %{y:.1f} %<extra>Y = 1</extra>"))
        aplicar_estilo(fig, "Distribución por grupo", f"{nombre} ({unidad})", "% de personas del grupo", alto=380,
                       subtitulo=nombre)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig)
        st.caption(f"Mediana Y = 0: **{fmt_dec(dist['mediana_0'])} {unidad}** · "
                   f"Mediana Y = 1: **{fmt_dec(dist['mediana_1'])} {unidad}** · "
                   f"AUC univariado: **{fmt_dec(rank.loc[var, 'auc'], 3)}**")

    med_pantalla = distribucion_por_clase("daily_screen_time_hours")
    bloque_hallazgo(
        evidencia=f"AUC univariado: pantalla diaria **{fmt_dec(rank.loc['daily_screen_time_hours', 'auc'])}**, "
                  f"fin de semana **{fmt_dec(rank.loc['weekend_screen_time', 'auc'])}**, redes sociales "
                  f"**{fmt_dec(rank.loc['social_media_hours', 'auc'])}**. La mediana de pantalla diaria es "
                  f"{fmt_dec(med_pantalla['mediana_1'])} h en Y = 1 vs {fmt_dec(med_pantalla['mediana_0'])} h en Y = 0.",
        patron="Las distribuciones de estas tres variables están claramente desplazadas entre grupos; "
               "videojuegos y trabajo/estudio separan de forma moderada y el resto se superpone casi por completo.",
        interpretacion="El uso problemático está asociado principalmente a **cuánto** tiempo se usa la pantalla y a **en qué** "
        "(redes sociales), más que a quién es la persona.",
        modelado="Priorizar estas variables; cuidar su imputación (redes sociales tiene ≈ 19 % de nulos) "
                 "y considerar variables derivadas como la proporción de tiempo en redes sociales.",
    )

# --- Hallazgo 3: efecto umbral y relación no lineal ---
with tab3:
    st.subheader("Hallazgo 3: la relación no es lineal, sino de tipo umbral (curva en S)")

    col_izq, col_der = st.columns(2)
    with col_izq:
        var = st.selectbox(
            "Variable para ver cómo cambia la probabilidad de Y = 1:",
            list(NUMERICAS), index=0, format_func=lambda c: NUMERICAS[c][0], key="sel_h3",
        )
        nombre, unidad, ancho = NUMERICAS[var]
        curva = tasa_por_tramo(var)
        fig = go.Figure(go.Scatter(
            x=curva["centro"], y=curva["tasa"] * 100, mode="lines+markers", name="% con Y = 1",
            line=dict(color=NARANJO, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
            customdata=np.stack([curva["inicio"], curva["inicio"] + ancho, curva["n"]], axis=-1),
            hovertemplate="Tramo [%{customdata[0]:g} – %{customdata[1]:g})<br>"
                          "Y = 1: %{y:.1f} %<br>n = %{customdata[2]:,.0f}<extra></extra>",
        ))
        linea_promedio(fig, res["tasa_y"])
        if var == "daily_screen_time_hours":
            fig.add_vrect(x0=inicio_subida, x1=fin_umbral, fillcolor=GRIS, opacity=0.2, line_width=0, layer="below",
                          annotation_text="Zona de transición", annotation_position="bottom right",
                          annotation_font=dict(color="#898781", size=11))
        aplicar_estilo(fig, "% con uso problemático por tramo", f"{nombre} ({unidad})",
                       "% con uso problemático (Y = 1)", alto=420, subtitulo=nombre)
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(fig)
        st.caption(f"Tramos de {fmt_dec(ancho, 1 if ancho < 1 else 0)} {unidad}; se omiten tramos con menos de "
                   f"{MIN_OBS_TRAMO} personas. Prueba con *Horas de sueño* o *Aperturas de apps* para comparar.")

    with col_der:
        st.markdown("&nbsp;")
        st.markdown("&nbsp;")
        fig = go.Figure(go.Heatmap(
            z=mapa_tasa.to_numpy() * 100, x=list(mapa_tasa.columns), y=list(mapa_tasa.index),
            colorscale=escala_secuencial(), zmin=0, zmax=100,
            customdata=mapa_n.to_numpy(), xgap=2, ygap=2,
            colorbar=dict(title="% Y = 1", ticksuffix=" %"),
            hovertemplate="Pantalla %{y} · Redes %{x}<br>Y = 1: %{z:.1f} %<br>n = %{customdata:,.0f}<extra></extra>",
        ))
        for fila in mapa_tasa.index:
            for columna in mapa_tasa.columns:
                valor = mapa_tasa.loc[fila, columna]
                if pd.notna(valor):
                    fig.add_annotation(x=columna, y=fila, text=f"{valor * 100:.0f} %", showarrow=False,
                                       font=dict(color=color_texto_celda(valor), size=12))
        aplicar_estilo(fig, "% con uso problemático", "Horas diarias en redes sociales", "Horas diarias de pantalla",
                       alto=420, subtitulo="Según horas diarias de pantalla y de redes sociales")
        st.plotly_chart(fig)
        st.caption("Celdas vacías: menos de 100 personas en esa combinación.")

    bajo_sin_redes = mapa_tasa.iloc[0, 0]
    bajo_con_redes = mapa_tasa.iloc[0, 2]
    bloque_hallazgo(
        evidencia=f"Bajo 4 h diarias la tasa de Y = 1 se mantiene entre {fmt_pct(tasa_bajo_4h.min(), 0)} y "
                  f"{fmt_pct(tasa_bajo_4h.max(), 0)}; cruza el 50 % cerca de las {fmt_dec(inicio_umbral, 1)} h y "
                  f"supera el 95 % desde las {fmt_dec(fin_umbral, 1)} h. Con menos de 4 h de pantalla, quien usa "
                  f"< 1 h de redes tiene {fmt_pct(bajo_sin_redes, 0)} de Y = 1, y quien usa 2–3 h, {fmt_pct(bajo_con_redes, 0)}.",
        patron="Meseta baja → subida brusca → saturación cercana al 100 %. Sobre las 10 h la etiqueta es "
               "prácticamente siempre 1, sin importar el resto de variables.",
        interpretacion="Una correlación lineal subestima la relación: el efecto depende del rango y **interactúa** con las redes "
        "sociales (que importan sobre todo cuando el tiempo total de pantalla es bajo o medio).",
        modelado="Preferir modelos que capturen no linealidades e interacciones (árboles, *random forest*, "
                 "*gradient boosting*) o, si se usa regresión logística, incluir tramos/splines e interacciones.",
    )

# --- Hallazgo 4: variables de contexto sin señal ---
with tab4:
    st.subheader("Hallazgo 4: el contexto personal y el sueño casi no se relacionan con Y")

    col_izq, col_der = st.columns(2)
    with col_izq:
        var = st.selectbox(
            "Variable categórica de contexto:",
            list(CATEGORICAS), index=1, format_func=lambda c: CATEGORICAS[c][0], key="sel_h4",
        )
        nombre = CATEGORICAS[var][0]
        t = tasa_por_categoria(var)
        fig = go.Figure(go.Bar(
            x=t.index, y=t["tasa"] * 100, marker=dict(color=NARANJO, cornerradius=4), width=0.5,
            text=[fmt_pct(v) for v in t["tasa"]], textposition="inside", insidetextanchor="middle", textangle=0,
            textfont=dict(color="#ffffff", size=14),
            customdata=t["n"], hovertemplate="%{x}<br>Y = 1: %{y:.1f} %<br>n = %{customdata:,.0f}<extra></extra>",
        ))
        linea_promedio(fig, res["tasa_y"])
        aplicar_estilo(fig, "% con uso problemático por categoría", nombre, "% con uso problemático (Y = 1)", alto=420,
                       subtitulo=nombre)
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(fig)
        st.caption(f"V de Cramér con Y: **{fmt_dec(rank.loc[var, 'cramer_v'], 3)}** (0 = sin asociación, 1 = asociación total).")

    with col_der:
        st.markdown("&nbsp;")
        st.markdown("&nbsp;")
        umbral = umbral_por_estres()
        fig = go.Figure()
        for (nivel, etiqueta), color in zip(CATEGORICAS["stress_level"][1].items(), RAMPA_ORDINAL):
            sub = umbral[umbral["stress_level"] == nivel]
            fig.add_trace(go.Scatter(
                x=sub["centro"], y=sub["tasa"] * 100, mode="lines", name=f"Estrés {etiqueta.lower()}",
                line=dict(color=color, width=2),
                hovertemplate="%{x:.1f} h: %{y:.1f} %<extra>Estrés " + etiqueta.lower() + "</extra>",
            ))
        aplicar_estilo(fig, "Efecto de la pantalla según el estrés", "Tiempo de pantalla diario (h/día)",
                       "% con uso problemático (Y = 1)", alto=420,
                       subtitulo="Si el estrés influyera, las curvas se separarían")
        fig.update_yaxes(range=[0, 105])
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig)
        st.caption("Las tres curvas se superponen: el estrés no modifica el efecto del tiempo de pantalla.")

    bloque_hallazgo(
        evidencia=f"La tasa de Y = 1 es {fmt_pct(tasas_estres.min())}–{fmt_pct(tasas_estres.max())} en todos los niveles "
                  f"de estrés y {fmt_pct(tasas_genero.min())}–{fmt_pct(tasas_genero.max())} entre géneros "
                  f"(V de Cramér ≤ {fmt_dec(v_max_categoricas, 3)}). Edad, sueño, notificaciones y aperturas de apps "
                  f"tienen AUC entre {fmt_dec(auc_debiles.min())} y {fmt_dec(auc_debiles.max())}.",
        patron="Barras planas alrededor del promedio global y curvas superpuestas. Notificaciones y aperturas de apps "
               "muestran oscilaciones entre tramos, pero sin una tendencia interpretable.",
        interpretacion="Esperábamos que el estrés, el sueño y el impacto académico se asociaran al uso problemático; en estos datos "
        "no ocurre. La señal está en los **hábitos de uso**, no en el contexto personal.",
        modelado="Candidatas a eliminarse o a evaluarse con importancia por permutación. Mantenerlas sin evidencia "
                 "agrega ruido y dimensionalidad (sobre todo al codificar las categóricas).",
    )

# --- Hallazgo 5: contexto semanal T ---
with tab5:
    st.subheader("Hallazgo 5: el fin de semana sube el uso por igual en ambos grupos (T no los distingue)")

    col_izq, col_der = st.columns(2)
    with col_izq:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=finde["centros"], y=finde["pct_0"], mode="lines", name="Y = 0 (sin uso problemático)",
                                 line=dict(color=AZUL, width=2), hovertemplate="%{x:+.2f} h: %{y:.1f} %<extra>Y = 0</extra>"))
        fig.add_trace(go.Scatter(x=finde["centros"], y=finde["pct_1"], mode="lines", name="Y = 1 (con uso problemático)",
                                 line=dict(color=NARANJO, width=2), hovertemplate="%{x:+.2f} h: %{y:.1f} %<extra>Y = 1</extra>"))
        fig.add_vline(x=0, line=dict(color="#898781", width=1, dash="dot"),
                      annotation_text="Sin cambio", annotation_position="top left",
                      annotation_font=dict(color="#898781", size=11))
        aplicar_estilo(fig, "Cambio de uso en fin de semana", "Pantalla fin de semana − pantalla diaria (h)",
                       "% de personas del grupo", alto=420, subtitulo="Respecto a un día hábil, por grupo")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig)

    with col_der:
        fig = go.Figure(go.Heatmap(
            z=np.where(finde["mapa"] > 0, finde["mapa"], np.nan), x=finde["bx"], y=finde["by"],
            colorscale=escala_secuencial(), colorbar=dict(title="Personas"),
            hovertemplate="Diario ≈ %{x} h · Fin de semana ≈ %{y} h<br>%{z:,.0f} personas<extra></extra>",
        ))
        fig.add_trace(go.Scatter(x=[0, 15], y=[0, 15], mode="lines", line=dict(color="#898781", width=1, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=14, y=13, text="Igual uso", showarrow=False, font=dict(color="#898781", size=11))
        aplicar_estilo(fig, "Pantalla diaria vs fin de semana", "Tiempo de pantalla diario (h/día)",
                       "Tiempo de pantalla fin de semana (h/día)", alto=420,
                       subtitulo=f"Correlación r = {fmt_dec(finde['corr'])}")
        st.plotly_chart(fig)

    bloque_hallazgo(
        evidencia=f"El fin de semana se usa en promedio {fmt_dec(finde['dif_media'])} h más que un día hábil: "
                  f"+{fmt_dec(finde['media_0'])} h en Y = 0 y +{fmt_dec(finde['media_1'])} h en Y = 1. "
                  f"La correlación entre ambas medidas es r = {fmt_dec(finde['corr'])}.",
        patron="Las dos curvas del cambio semanal son prácticamente idénticas y la nube se ubica sobre la diagonal "
               "de forma paralela: quien usa más en la semana también usa más el fin de semana.",
        interpretacion="Dentro de la semana típica (T), el fin de semana no revela un comportamiento distinto del uso problemático; "
        "la información útil está en el **nivel** de uso, no en su variación semanal.",
        modelado="Tratar ambas variables como redundantes: conservar una, promediarlas o regularizar. "
                 "La diferencia fin de semana − diario no parece aportar como variable derivada.",
    )

st.markdown("---")

# --- Limitaciones ---
st.header("Limitaciones detectadas en los datos")
st.markdown(f"""
* **Muchos faltantes repartidos:** solo el {fmt_pct(res["filas_completas"])} de las filas está completa, por lo que eliminar filas con nulos descartaría la mayoría de la muestra. A favor: los nulos **no parecen informativos**, ya que la tasa de Y = 1 en filas con nulo es {fmt_pct(calidad["tasa_nulos_min"])}–{fmt_pct(calidad["tasa_nulos_max"])}, igual a la global (diferencia máxima de {fmt_dec(calidad["dif_max_nulos"] * 100)} p.p.). Esto respalda la imputación simple propuesta en *Problema y Datos*.
* **Posible circularidad de la etiqueta:** sobre las 10 h diarias prácticamente el 100 % está etiquetado como adicto. Es posible que `addicted_label` se haya construido a partir del tiempo de pantalla; si es así, el modelo podría limitarse a "redescubrir" esa regla.
* **Indicios de datos sintéticos:** la ausencia total de relación con estrés y sueño es poco realista, y hay valores exactos muy repetidos (por ejemplo, {fmt_num(calidad["valor_aperturas_top"])} aperturas de apps aparece {fmt_num(calidad["n_aperturas_top"])} veces). Las conclusiones podrían no generalizarse a población real.
* **Coherencia interna adecuada:** en el {fmt_pct(calidad["pct_suma_coherente"])} de los casos completos el tiempo de pantalla diario es mayor o igual que la suma de redes sociales, videojuegos y trabajo/estudio (r = {fmt_dec(calidad["corr_suma"])}), lo que puede aprovecharse para imputar con más información.
* **Diseño transversal:** T es una semana típica; no hay seguimiento en el tiempo, por lo que no se puede estudiar la evolución del uso ni inferir causalidad.
""")

# --- Implicancias para el modelado ---
st.header("¿Qué implica esto para el Avance 3?")
implicancias = pd.DataFrame({
    "Hallazgo": [
        "H1 · Y desbalanceada",
        "H2 · Predictores clave",
        "H3 · Efecto umbral e interacción",
        "H4 · Contexto sin señal",
        "H5 · Semana vs fin de semana",
        "Limitaciones",
    ],
    "Decisión para el modelado": [
        "Partición estratificada; métricas F1, ROC-AUC y PR-AUC; probar class_weight.",
        "Centrar el modelo en tiempo de pantalla y redes sociales; imputar con cuidado.",
        "Modelos no lineales (árboles / boosting) o logística con tramos e interacciones.",
        "Evaluar eliminar estrés, género, impacto, edad, sueño, notificaciones y aperturas.",
        "Evitar usar ambas medidas de pantalla sin regularización (colinealidad r ≈ 0,8).",
        "Imputación dentro del pipeline; contrastar el modelo contra una regla simple de horas de pantalla.",
    ],
})
st.dataframe(implicancias, hide_index=True)

st.info(
    "**¿Nuestra pregunta sigue teniendo sentido?** Sí, con un matiz: los hábitos digitales (especialmente el tiempo "
    "de pantalla y las redes sociales) permiten distinguir con claridad el uso problemático, pero el contexto personal "
    "(estrés, impacto académico, género) y el sueño aportan muy poco. El Avance 3 debería verificar si un modelo supera "
    "de forma relevante a una regla simple basada en horas de pantalla."
)
