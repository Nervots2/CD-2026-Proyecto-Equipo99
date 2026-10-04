"""Constantes, carga de datos y utilidades de presentación compartidas por las páginas de la app."""
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Constantes del proyecto ---
Y = "addicted_label"

PREGUNTA = (
    "¿Es posible identificar si una persona de 18 a 35 años presenta un patrón de uso problemático de "
    "pantallas ($Y$), a partir de sus hábitos digitales, de sueño y de su contexto personal medidos en una "
    "semana típica ($X$), usando únicamente la información disponible al momento de la evaluación ($T$)?"
)

# Rutas posibles del dataset (estructura data/raw del repositorio o raíz del proyecto)
RAIZ = Path(__file__).resolve().parents[1]
RUTAS_DATOS = [
    RAIZ / "data" / "raw" / "adiccion_pantalla.csv",
    RAIZ / "adiccion_pantalla.csv",
]

# Variables numéricas: (nombre legible, unidad, ancho del tramo para agrupar)
NUMERICAS = {
    "daily_screen_time_hours": ("Pantalla diaria", "h/día", 0.5),
    "weekend_screen_time": ("Pantalla fin de semana", "h/día", 0.5),
    "social_media_hours": ("Redes sociales", "h/día", 0.5),
    "gaming_hours": ("Videojuegos", "h/día", 0.25),
    "work_study_hours": ("Trabajo o estudio", "h/día", 0.5),
    "sleep_hours": ("Sueño", "h/noche", 0.25),
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

COMPONENTES = ["social_media_hours", "gaming_hours", "work_study_hours"]

MIN_OBS_TRAMO = 200  # tramos con menos observaciones se omiten para no graficar ruido

GRIS = "#898781"
ESCALA_SECUENCIAL = [
    [0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"],
]


# --- Formato de números (estilo chileno: 691.369 y 70,9 %) ---
def fmt_num(x):
    return f"{x:,.0f}".replace(",", ".")


def fmt_dec(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


def fmt_pct(x, dec=1):
    return f"{x * 100:.{dec}f}".replace(".", ",") + " %"


def nombre(col):
    return NUMERICAS[col][0] if col in NUMERICAS else CATEGORICAS[col][0]


def etiqueta(col):
    """Nombre legible con su unidad, para los ejes."""
    if col in NUMERICAS:
        return f"{NUMERICAS[col][0]} ({NUMERICAS[col][1]})"
    return CATEGORICAS[col][0]


# --- Carga de datos ---
# cache_resource entrega siempre el mismo DataFrame (sin copiarlo): ninguna función debe modificarlo
@st.cache_resource(show_spinner="Cargando datos…")
def cargar_datos():
    for ruta in RUTAS_DATOS:
        if ruta.exists():
            return pd.read_csv(ruta)
    return None


def exigir_datos():
    """Detiene la página con un mensaje claro si no se encuentra el dataset."""
    if cargar_datos() is None:
        st.error("No se encontró `adiccion_pantalla.csv`. Colócalo en `data/raw/` o en la raíz del proyecto.")
        st.stop()


@st.cache_data
def resumen_general():
    df = cargar_datos()
    return {
        "n": len(df),
        "n_vars": df.shape[1],
        "tasa_y": df[Y].mean(),
        "n_y1": int(df[Y].sum()),
        "filas_completas": df.dropna().shape[0] / len(df),
    }


# --- Utilidades de presentación ---
def tema_oscuro():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def colores():
    """Colores de las series según el tema: el azul es siempre Y = 0 y el naranjo Y = 1."""
    if tema_oscuro():
        return {"y0": "#3987e5", "y1": "#d95926", "fondo": "#0e1117"}
    return {"y0": "#2a78d6", "y1": "#eb6834", "fondo": "#ffffff"}


def escala_secuencial():
    # En modo oscuro se invierte la rampa para que los valores bajos se confundan con el fondo
    if tema_oscuro():
        return [[pos, color] for pos, (_, color) in zip([p for p, _ in ESCALA_SECUENCIAL], reversed(ESCALA_SECUENCIAL))]
    return ESCALA_SECUENCIAL


def color_texto_celda(proporcion):
    """Texto oscuro sobre celdas claras y blanco sobre celdas oscuras, según el tema activo."""
    celda_oscura = (proporcion > 0.5) != tema_oscuro()
    return "#ffffff" if celda_oscura else "#0b0b0b"


def aplicar_estilo(fig, titulo, eje_x, eje_y, alto=420, subtitulo=None, leyenda=True):
    fig.update_layout(
        # Título fijo arriba del contenedor para dejar espacio a la leyenda
        title=dict(text=titulo, font=dict(size=16), subtitle=dict(text=subtitulo or ""),
                   yref="container", y=1, yanchor="top", pad=dict(t=10)),
        xaxis_title=eje_x,
        yaxis_title=eje_y,
        height=alto,
        separators=",.",
        margin=dict(t=120 if leyenda else 80, l=10, r=10, b=10),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


def linea_promedio(fig, tasa, con_texto=True):
    """Línea punteada de referencia con el % global de Y = 1."""
    anotacion = {}
    if con_texto:
        anotacion = dict(annotation_text=f"Promedio global: {fmt_pct(tasa)}", annotation_position="top left",
                         annotation_font=dict(color=GRIS, size=11))
    fig.add_hline(y=tasa * 100, line=dict(color=GRIS, width=1, dash="dot"), **anotacion)


def tabla_datos(tabla):
    """Los mismos datos del gráfico en formato de tabla."""
    with st.expander("Ver datos del gráfico"):
        st.dataframe(tabla, hide_index=True)
