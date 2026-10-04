import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utilidades import (
    CATEGORICAS, GRIS, MIN_OBS_TRAMO, NUMERICAS, PREGUNTA, Y,
    aplicar_estilo, cargar_datos, color_texto_celda, colores, escala_secuencial, etiqueta, exigir_datos,
    fmt_dec, fmt_num, fmt_pct, linea_promedio, nombre, resumen_general, tabla_datos,
)

# Configuración básica de la página
st.set_page_config(page_title="Hallazgos", page_icon="🔎", layout="wide")

GRUPOS = {0: "Y = 0 (sin uso problemático)", 1: "Y = 1 (con uso problemático)"}

TRAMOS_PANTALLA = ([0, 4, 6, 8, 10, 24], ["< 4 h", "4–6 h", "6–8 h", "8–10 h", "≥ 10 h"])
TRAMOS_REDES = ([0, 1, 2, 3, 4, 24], ["< 1 h", "1–2 h", "2–3 h", "3–4 h", "≥ 4 h"])

# Grupos de contexto para el hallazgo 4 (las numéricas se agrupan en tramos)
TRAMOS_EDAD = ([17, 22, 27, 31, 35], ["18–22 años", "23–27 años", "28–31 años", "32–35 años"])
TRAMOS_SUENO = ([4.5, 5.5, 6.5, 7.5, 8.5, 9.01], ["< 5,5 h", "5,5–6,5 h", "6,5–7,5 h", "7,5–8,5 h", "≥ 8,5 h"])
CONTEXTO = {
    "stress_level": "Nivel de estrés",
    "sleep_hours": "Horas de sueño",
    "academic_work_impact": "Impacto académico/laboral",
    "gender": "Género",
    "age": "Grupo de edad",
}


# --- Cálculos (cacheados para no recalcular en cada interacción) ---
@st.cache_data
def tasa_por_tramo(col):
    """% de personas con Y = 1 en cada tramo de la variable."""
    df = cargar_datos()
    ancho = NUMERICAS[col][2]
    d = df[[col, Y]].dropna()
    tramo = np.floor(d[col] / ancho) * ancho
    t = d.groupby(tramo)[Y].agg(tasa="mean", n="size").reset_index(names="inicio")
    # La edad es un número entero: el punto va sobre el valor y no al centro del tramo
    t["centro"] = t["inicio"] + (0 if col == "age" else ancho / 2)
    return t[t["n"] >= MIN_OBS_TRAMO]


@st.cache_data
def medianas_por_grupo():
    df = cargar_datos()
    med = df.groupby(Y)[list(NUMERICAS)].median().T
    tabla = pd.DataFrame({
        "Variable": [etiqueta(c) for c in med.index],
        "Y = 0": med[0].to_numpy(),
        "Y = 1": med[1].to_numpy(),
    }, index=med.index)
    tabla["Diferencia"] = (tabla["Y = 1"] / tabla["Y = 0"] - 1) * 100
    return tabla.sort_values("Diferencia", ascending=False, key=abs)


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
def tasa_por_grupo(col):
    """% de personas con Y = 1 en cada grupo de una variable de contexto."""
    df = cargar_datos()
    d = df[[col, Y]].dropna()
    if col == "age":
        grupo = pd.cut(d[col], TRAMOS_EDAD[0], labels=TRAMOS_EDAD[1])
    elif col == "sleep_hours":
        grupo = pd.cut(d[col], TRAMOS_SUENO[0], labels=TRAMOS_SUENO[1], right=False)
    else:
        traduccion = CATEGORICAS[col][1]
        grupo = pd.Categorical(d[col].map(traduccion), categories=list(traduccion.values()))
    return d.groupby(grupo, observed=False)[Y].agg(tasa="mean", n="size")


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
        "pct_mas_finde": (dif > 0).mean(),
    }
    for clase in (0, 1):
        valores = dif[d[Y] == clase]
        conteo, _ = np.histogram(valores, bins=bordes)
        salida[f"pct_{clase}"] = conteo / len(valores) * 100
        salida[f"media_{clase}"] = valores.mean()
        salida[f"pantalla_{clase}"] = d.loc[d[Y] == clase, "daily_screen_time_hours"].mean()
    return salida


@st.cache_data
def limitaciones():
    df = cargar_datos()
    sobre_10h = df.loc[df["daily_screen_time_hours"] >= 10, Y]
    return {"n_10h": len(sobre_10h), "n_10h_y0": int((sobre_10h == 0).sum()), "tasa_10h": sobre_10h.mean()}


# --- Utilidades de presentación ---
def bloque_hallazgo(evidencia, patron, interpretacion):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**📊 Evidencia**\n\n{evidencia}")
    with c2:
        st.markdown(f"**🔁 Patrón**\n\n{patron}")
    with c3:
        st.markdown(f"**💡 Interpretación y relación con la pregunta**\n\n{interpretacion}")


# --- Encabezado ---
st.title("🔎 Hallazgos")
st.markdown("---")
exigir_datos()

res = resumen_general()
color = colores()
medianas = medianas_por_grupo()
mapa_tasa, mapa_n = mapa_pantalla_redes()
finde = semana_vs_finde()
lim = limitaciones()
curva_pantalla = tasa_por_tramo("daily_screen_time_hours")

# Valores clave del efecto umbral, calculados desde los datos
inicio_subida = curva_pantalla.loc[curva_pantalla["tasa"] >= 0.4, "inicio"].min()
inicio_umbral = curva_pantalla.loc[curva_pantalla["tasa"] >= 0.5, "inicio"].min()
fin_umbral = curva_pantalla.loc[curva_pantalla["tasa"] >= 0.95, "inicio"].min()
tasa_bajo_4h = curva_pantalla.loc[curva_pantalla["inicio"] < 4, "tasa"]
rangos_contexto = {col: tasa_por_grupo(col)["tasa"] for col in CONTEXTO}

st.header("¿Qué hemos aprendido de nuestros datos?")
st.markdown(
    "Cada hallazgo sigue la secuencia **Evidencia → Patrón → Interpretación** y se conecta con la pregunta del proyecto:"
)
st.info(f"**Pregunta:** {PREGUNTA}")

# --- Resumen explícito de los hallazgos ---
st.subheader("Resumen de hallazgos")
st.markdown(f"""
1. **La variable objetivo está desbalanceada:** {fmt_pct(res["tasa_y"])} de las personas presenta uso problemático (≈ 7 de cada 10).
2. **El tiempo de pantalla y las redes sociales son las variables que más informan sobre $Y$**, con mucha diferencia sobre el resto.
3. **La relación no es lineal:** sigue un efecto umbral (curva en S) y, bajo ese umbral, las redes sociales marcan la diferencia.
4. **El contexto personal y el sueño casi no se relacionan con $Y$**, un resultado que no esperábamos según nuestra pregunta inicial.
5. **El contexto semanal ($T$) no distingue grupos:** todos aumentan su uso el fin de semana en la misma magnitud.
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
    m1, m2, m3 = st.columns(3)
    m1.metric("Con uso problemático (Y = 1)", fmt_pct(res["tasa_y"]), help=f"{fmt_num(res['n_y1'])} personas")
    m2.metric("Sin uso problemático (Y = 0)", fmt_pct(1 - res["tasa_y"]), help=f"{fmt_num(n_y0)} personas")
    m3.metric("Razón entre clases", f"{fmt_dec(res['n_y1'] / n_y0, 1)} : 1")
    st.caption("El gráfico de la distribución de Y está en la página **Análisis Exploratorio**.")

    bloque_hallazgo(
        evidencia=f"{fmt_num(res['n_y1'])} personas con Y = 1 ({fmt_pct(res['tasa_y'])}) frente a "
                  f"{fmt_num(n_y0)} con Y = 0. La etiqueta no tiene valores faltantes.",
        patron=f"La clase *con uso problemático* es la mayoritaria, en proporción ≈ {fmt_dec(res['n_y1'] / n_y0, 1)} : 1.",
        interpretacion=f"Un modelo que siempre prediga \"uso problemático\" ya acertaría el {fmt_pct(res['tasa_y'])} "
                       "de las veces. Para responder la pregunta no basta con medir la exactitud: hay que revisar "
                       "qué tan bien se identifica a cada grupo.",
    )

# --- Hallazgo 2: predictores clave ---
with tab2:
    st.subheader("Hallazgo 2: el tiempo de pantalla y las redes sociales concentran la información sobre Y")

    col_izq, col_der = st.columns([3, 2])
    with col_izq:
        var = st.selectbox(
            "Variable para ver cómo cambia el % de uso problemático:",
            list(NUMERICAS), index=0, format_func=nombre, key="sel_h2",
        )
        unidad, ancho = NUMERICAS[var][1], NUMERICAS[var][2]
        curva = tasa_por_tramo(var)
        fig = go.Figure(go.Scatter(
            x=curva["centro"], y=curva["tasa"] * 100, mode="lines+markers", name="% con Y = 1",
            line=dict(color=color["y1"], width=2), marker=dict(size=8, line=dict(color=color["fondo"], width=2)),
            customdata=np.stack([curva["inicio"], curva["inicio"] + ancho, curva["n"]], axis=-1),
            hovertemplate="Tramo [%{customdata[0]:g} – %{customdata[1]:g})<br>"
                          "Y = 1: %{y:.1f} %<br>n = %{customdata[2]:,.0f}<extra></extra>",
        ))
        linea_promedio(fig, res["tasa_y"])
        if var == "daily_screen_time_hours":
            fig.add_vrect(x0=inicio_subida, x1=fin_umbral, fillcolor=GRIS, opacity=0.15, line_width=0, layer="below",
                          annotation_text="Zona de transición", annotation_position="bottom right",
                          annotation_font=dict(color=GRIS, size=11))
        aplicar_estilo(fig, "% con uso problemático por tramo", etiqueta(var), "% con uso problemático (Y = 1)",
                       alto=420, subtitulo=nombre(var), leyenda=False)
        fig.update_yaxes(range=[0, 105], ticksuffix=" %")
        st.plotly_chart(fig)
        st.caption(f"Tramos de {fmt_dec(ancho, 2 if ancho < 0.5 else 1 if ancho < 1 else 0)} {unidad}; se omiten los "
                   f"tramos con menos de {MIN_OBS_TRAMO} personas. Compara *Pantalla diaria* con *Sueño* o *Edad*.")
        tabla_datos(pd.DataFrame({
            f"Inicio del tramo ({unidad})": curva["inicio"],
            "% con Y = 1": curva["tasa"] * 100,
            "Personas": curva["n"],
        }).style.format({"% con Y = 1": "{:.1f}", f"Inicio del tramo ({unidad})": "{:g}"}, decimal=",", thousands="."))

    with col_der:
        st.markdown("**Mediana de cada variable según el grupo**")
        st.dataframe(
            medianas.style.format({"Y = 0": "{:.2f}", "Y = 1": "{:.2f}", "Diferencia": "{:+.0f} %"}, decimal=","),
            hide_index=True, height=35 * (len(medianas) + 1) + 3,
        )
        st.caption("Diferencia de la mediana de Y = 1 respecto de la de Y = 0, de mayor a menor.")

    bloque_hallazgo(
        evidencia=f"La mediana de pantalla diaria es **{fmt_dec(medianas.loc['daily_screen_time_hours', 'Y = 1'])} h** "
                  f"en Y = 1 frente a **{fmt_dec(medianas.loc['daily_screen_time_hours', 'Y = 0'])} h** en Y = 0; "
                  f"en redes sociales, {fmt_dec(medianas.loc['social_media_hours', 'Y = 1'])} h frente a "
                  f"{fmt_dec(medianas.loc['social_media_hours', 'Y = 0'])} h. El % de uso problemático pasa de "
                  f"≈ {fmt_pct(tasa_bajo_4h.mean(), 0)} con poco uso a casi 100 % con uso alto.",
        patron="Pantalla diaria, fin de semana y redes sociales tienen curvas que suben con fuerza. Videojuegos y "
               "trabajo o estudio suben de forma moderada. Sueño, edad, notificaciones y aperturas se mantienen "
               "planas alrededor del promedio.",
        interpretacion="Responde a la pregunta: **sí** es posible identificar el uso problemático a partir de los "
                       "hábitos digitales, principalmente **cuánto** se usa la pantalla y **en qué** (redes sociales).",
    )

# --- Hallazgo 3: efecto umbral y relación no lineal ---
with tab3:
    st.subheader("Hallazgo 3: la relación no es lineal, sino de tipo umbral, y depende de las redes sociales")

    col_izq, col_der = st.columns([3, 2])
    with col_izq:
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
                                       font=dict(color=color_texto_celda(valor), size=13))
        aplicar_estilo(fig, "% con uso problemático según pantalla y redes sociales",
                       "Horas diarias en redes sociales", "Horas diarias de pantalla", alto=440,
                       subtitulo="Cada celda es una combinación de tiempo total y tiempo en redes", leyenda=False)
        st.plotly_chart(fig)
        st.caption("Celdas vacías: menos de 100 personas en esa combinación.")
        tabla_datos((mapa_tasa * 100).rename_axis("Pantalla / Redes").reset_index().style.format(
            precision=1, decimal=",", na_rep="—"))

    with col_der:
        st.markdown("**Cómo leer el gráfico**")
        st.markdown(f"""
* **De abajo hacia arriba (más pantalla):** el % de uso problemático se mantiene bajo, sube de golpe entre las {fmt_dec(inicio_subida, 1)} y las {fmt_dec(fin_umbral, 1)} h y luego se satura cerca del 100 %. Es la curva en S de la pestaña H2.
* **De izquierda a derecha (más redes):** dentro de una misma fila el % también sube, sobre todo cuando el tiempo total es bajo o medio.
* **Fila superior (≥ 10 h):** casi todos tienen uso problemático, sin importar las redes.
""")

    bajo_sin_redes = mapa_tasa.iloc[0, 0]
    bajo_con_redes = mapa_tasa.iloc[0, 2]
    bloque_hallazgo(
        evidencia=f"Bajo 4 h diarias de pantalla el % de Y = 1 se mantiene entre {fmt_pct(tasa_bajo_4h.min(), 0)} y "
                  f"{fmt_pct(tasa_bajo_4h.max(), 0)}; cruza el 50 % cerca de las {fmt_dec(inicio_umbral, 1)} h y "
                  f"supera el 95 % desde las {fmt_dec(fin_umbral, 1)} h. Con menos de 4 h de pantalla, quien usa "
                  f"< 1 h de redes tiene {fmt_pct(bajo_sin_redes, 0)} de Y = 1, y quien usa 2–3 h, {fmt_pct(bajo_con_redes, 0)}.",
        patron="Meseta baja → subida brusca → saturación cercana al 100 %. Además, el efecto de las redes sociales "
               "depende del tiempo total de pantalla (interacción).",
        interpretacion="El tiempo total no lo explica todo: con el mismo tiempo de pantalla, el **patrón de uso** "
                       "(cuánto de ese tiempo es redes sociales) cambia el resultado. Una relación lineal simple "
                       "no describe bien el fenómeno.",
    )

# --- Hallazgo 4: variables de contexto sin señal ---
with tab4:
    st.subheader("Hallazgo 4: el contexto personal y el sueño casi no se relacionan con Y")

    col_izq, col_der = st.columns([3, 2])
    with col_izq:
        var = st.selectbox(
            "Variable de contexto:", list(CONTEXTO), index=0, format_func=lambda c: CONTEXTO[c], key="sel_h4",
        )
        t = tasa_por_grupo(var)
        fig = go.Figure(go.Bar(
            x=list(t.index), y=t["tasa"] * 100, marker=dict(color=color["y1"], cornerradius=4), width=0.35,
            text=[fmt_pct(v) for v in t["tasa"]], textposition="outside", textfont=dict(size=13),
            customdata=t["n"], hovertemplate="%{x}<br>Y = 1: %{y:.1f} %<br>n = %{customdata:,.0f}<extra></extra>",
        ))
        linea_promedio(fig, res["tasa_y"], con_texto=False)  # el texto chocaría con las etiquetas de las barras
        aplicar_estilo(fig, "% con uso problemático por grupo", CONTEXTO[var], "% con uso problemático (Y = 1)",
                       alto=420, leyenda=False,
                       subtitulo=f"{CONTEXTO[var]} · línea punteada: promedio global ({fmt_pct(res['tasa_y'])})")
        fig.update_yaxes(range=[0, 105], ticksuffix=" %")
        st.plotly_chart(fig)
        st.caption(f"Diferencia entre el grupo más alto y el más bajo: "
                   f"**{fmt_dec((t['tasa'].max() - t['tasa'].min()) * 100, 1)} puntos porcentuales**.")
        tabla_datos(pd.DataFrame({
            CONTEXTO[var]: list(t.index), "% con Y = 1": t["tasa"].to_numpy() * 100, "Personas": t["n"].to_numpy(),
        }).style.format({"% con Y = 1": "{:.1f}"}, decimal=",", thousands="."))

    with col_der:
        st.markdown("**% con uso problemático: grupo más bajo y más alto de cada variable**")
        st.dataframe(
            pd.DataFrame({
                "Variable": list(CONTEXTO.values()),
                "Mínimo": [r.min() * 100 for r in rangos_contexto.values()],
                "Máximo": [r.max() * 100 for r in rangos_contexto.values()],
                "Dif. (p.p.)": [(r.max() - r.min()) * 100 for r in rangos_contexto.values()],
            }).style.format({"Mínimo": "{:.1f} %", "Máximo": "{:.1f} %", "Dif. (p.p.)": "{:.1f}"}, decimal=","),
            hide_index=True,
        )
        st.caption(f"Como referencia, entre el tramo más bajo y el más alto de pantalla diaria la diferencia es de "
                   f"{fmt_dec((curva_pantalla['tasa'].max() - curva_pantalla['tasa'].min()) * 100, 0)} puntos porcentuales.")

    dif_max = max((r.max() - r.min()) for r in rangos_contexto.values())
    bloque_hallazgo(
        evidencia=f"El % de Y = 1 va de {fmt_pct(rangos_contexto['stress_level'].min())} a "
                  f"{fmt_pct(rangos_contexto['stress_level'].max())} entre niveles de estrés y de "
                  f"{fmt_pct(rangos_contexto['sleep_hours'].min())} a {fmt_pct(rangos_contexto['sleep_hours'].max())} "
                  f"entre tramos de sueño. Ninguna variable de contexto supera los {fmt_dec(dif_max * 100, 1)} p.p. de diferencia.",
        patron="Barras casi planas alrededor del promedio global en todas las variables de contexto.",
        interpretacion="Resultado inesperado: nuestra pregunta incluía el sueño y el contexto personal, y esperábamos "
                       "que más estrés o menos sueño acompañaran al uso problemático. En estos datos no ocurre: la "
                       "señal está en los **hábitos de uso**, no en quién es la persona.",
    )

# --- Hallazgo 5: contexto semanal T ---
with tab5:
    st.subheader("Hallazgo 5: el fin de semana sube el uso por igual en ambos grupos (T no los distingue)")

    col_izq, col_der = st.columns([3, 2])
    with col_izq:
        fig = go.Figure()
        for clase in (0, 1):
            fig.add_trace(go.Scatter(
                x=finde["centros"], y=finde[f"pct_{clase}"], mode="lines", name=GRUPOS[clase],
                line=dict(color=color[f"y{clase}"], width=2),
                hovertemplate="%{x:+.2f} h: %{y:.1f} %<extra>Y = " + str(clase) + "</extra>",
            ))
        fig.add_vline(x=0, line=dict(color=GRIS, width=1, dash="dot"),
                      annotation_text="Sin cambio", annotation_position="top left",
                      annotation_font=dict(color=GRIS, size=11))
        aplicar_estilo(fig, "Cambio de uso en fin de semana", "Pantalla fin de semana − pantalla día hábil (h)",
                       "% de personas del grupo", alto=420, subtitulo="Respecto de un día hábil, por grupo")
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(rangemode="tozero", ticksuffix=" %")
        st.plotly_chart(fig)
        tabla_datos(pd.DataFrame({
            "Diferencia (h, centro del tramo)": finde["centros"],
            "% en Y = 0": finde["pct_0"], "% en Y = 1": finde["pct_1"],
        }).style.format(precision=2, decimal=","))

    with col_der:
        st.metric("Aumento en fin de semana · Y = 0", f"+{fmt_dec(finde['media_0'])} h")
        st.metric("Aumento en fin de semana · Y = 1", f"+{fmt_dec(finde['media_1'])} h")
        st.metric("Usa más pantalla el fin de semana", fmt_pct(finde["pct_mas_finde"]))

    bloque_hallazgo(
        evidencia=f"El fin de semana se usa en promedio {fmt_dec(finde['dif_media'])} h más que un día hábil: "
                  f"+{fmt_dec(finde['media_0'])} h en Y = 0 y +{fmt_dec(finde['media_1'])} h en Y = 1. "
                  f"En un día hábil, en cambio, Y = 1 usa {fmt_dec(finde['pantalla_1'] - finde['pantalla_0'], 1)} h más que Y = 0.",
        patron="Las dos curvas del cambio semanal son prácticamente idénticas: quien usa más en la semana también "
               f"usa más el fin de semana (correlación r = {fmt_dec(finde['corr'])}).",
        interpretacion="Dentro de la semana típica ($T$), el uso problemático no aparece como un \"atracón\" de fin de "
                       "semana. Lo que distingue a los grupos es el **nivel** de uso en ambos momentos de la semana, "
                       "no su variación.",
    )

st.markdown("---")

# --- Limitaciones ---
st.header("Limitaciones detectadas en los datos")
st.markdown(f"""
* **Indicios de datos sintéticos:** edad, sueño, notificaciones y aperturas se reparten de forma casi uniforme, y no aparecen relaciones esperables (por ejemplo, entre sueño y tiempo de pantalla). Las conclusiones podrían no generalizarse a una población real.
* **Posible circularidad de la etiqueta:** desde las 10 h diarias de pantalla, el {fmt_pct(lim["tasa_10h"])} está etiquetado con uso problemático (solo {fmt_num(lim["n_10h_y0"])} de {fmt_num(lim["n_10h"])} personas tienen Y = 0). Es posible que `addicted_label` se haya construido a partir del tiempo de pantalla; en ese caso, un modelo solo "redescubriría" esa regla.
* **Muchos faltantes repartidos:** solo el {fmt_pct(res["filas_completas"])} de las filas está completa. A favor, los faltantes no se relacionan con $Y$.
* **Diseño transversal:** $T$ es una semana típica; no hay seguimiento en el tiempo, por lo que no se puede estudiar la evolución del uso ni inferir causalidad.
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
        "Partición estratificada; métricas F1, recall por clase y ROC-AUC, no solo exactitud.",
        "Centrar el modelo en tiempo de pantalla y redes sociales; cuidar su imputación.",
        "Modelos no lineales (árboles, random forest, gradient boosting) o logística con tramos e interacciones.",
        "Evaluar si estrés, género, impacto, edad y sueño se pueden eliminar para simplificar el modelo.",
        "Pantalla diaria y de fin de semana están muy correlacionadas: regularizar o usar una para imputar la otra.",
        "Imputar dentro del pipeline y comparar el modelo contra una regla simple basada en horas de pantalla.",
    ],
})
st.dataframe(implicancias, hide_index=True)

st.success(
    "**Respuesta preliminar a la pregunta:** sí, con un matiz. Los hábitos digitales (especialmente el tiempo de "
    "pantalla y las redes sociales) permiten distinguir con claridad el uso problemático, con una relación de tipo "
    "umbral. En cambio, el sueño y el contexto personal (estrés, impacto académico o laboral, género y edad) aportan "
    "muy poco. La pregunta sigue teniendo sentido, pero el Avance 3 deberá verificar que un modelo supere de forma "
    "relevante a una regla simple basada en horas de pantalla."
)
