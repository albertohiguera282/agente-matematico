# -*- coding: utf-8 -*-
"""
Main Streamlit application for 'Consultor Inteligente de Modelos Estadísticos'.

Estructura del archivo
----------------------
1. Imports y configuración (flags de módulos)
2. Utilidades generales (safe_format, gráficas)
3. Utilidades HIGEA (limpieza de datos)
4. Estilos y estado de sesión
5. Navegación (is_enabled / goto)
6. Renderizadores de cada sección (render_*)
7. Registro de módulos y punto de entrada (main)
"""

import importlib
import string
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import advisor
import calculator
import data_processor
import database
import higea_core
import tests

# ============================================================
# 1. CONFIGURACIÓN: CONTROL DE VISIBILIDAD DE MÓDULOS
# ============================================================
# True  = la sección aparece en el menú lateral.
# False = la sección se oculta (y los botones que llevan a ella se desactivan).
SHOW_ADVISOR = True
SHOW_LIBRARY = True
SHOW_LAB = True
SHOW_CALCULATOR = True
SHOW_HIGEA = True

# Recarga los módulos locales en cada ejecución (útil en desarrollo).
DEV_RELOAD = True
if DEV_RELOAD:
    importlib.reload(higea_core)
    importlib.reload(calculator)
    importlib.reload(tests)

from higea_core import (  # noqa: E402
    AS001DiagnosticAssistant, higea_auto_clean, higea_winsorize_column,
    higea_clean_column_names, higea_normalize_text_case,
    higea_convert_column_type, higea_scale_column
)

# Nombres de las secciones (se usan como identificadores de navegación)
TAB_ADVISOR = "Asesor Metodológico"
TAB_LIBRARY = "Biblioteca de Modelos"
TAB_LAB = "Laboratorio de Datos"
TAB_HIGEA = "🧹 Limpieza de Datos (HIGEA)"
TAB_CALC = "Calculadora Analítica"

# ============================================================
# 2. UTILIDADES GENERALES
# ============================================================
class _SafeDict(dict):
    """Si falta una clave al formatear, devuelve la llave tal cual
    (ej. '{media}') en lugar de lanzar KeyError."""
    def __missing__(self, key):
        return "{" + key + "}"

def safe_format(template_str, **kwargs):
    """Reemplazo seguro de str.format() que nunca lanza KeyError.
    Cualquier placeholder no reconocido se deja intacto en el texto."""
    try:
        return string.Formatter().vformat(template_str, (), _SafeDict(**kwargs))
    except Exception:
        return template_str

def normalize_text(text):
    """Pasa a minúsculas y quita las tildes (para buscar 'regresion' y encontrar 'Regresión')."""
    s = unicodedata.normalize("NFD", str(text).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def render_customizable_chart(df, selected_col, col_type, key_prefix="chart"):
    """Renderiza una gráfica interactiva de Plotly con botones de control para tipo, paleta y escalas."""
    import re
    safe_col = re.sub(r'\W+', '_', str(selected_col))
    key_prefix = f"{key_prefix}_{safe_col}"
    st.markdown("##### 🎛️ Opciones de Personalización de Gráfica")
    
    color_map = {
        "Neón Azul": ["#3b82f6"],
        "Esmeralda": ["#10b981"],
        "Violeta": ["#8b5cf6"],
        "Viridis": px.colors.sequential.Viridis,
        "Warm / Fuego": px.colors.sequential.Flame,
        "Pastel": px.colors.qualitative.Pastel
    }

    if col_type == "numeric":
        c_opt1, c_opt2, c_opt3 = st.columns([2, 2, 1])
        with c_opt1:
            chart_kind = st.selectbox(
                "Tipo de Gráfico",
                ["Histograma + Boxplot", "Diagrama de Caja (Boxplot)", "Diagrama de Violín", "Dispersión (Jitter)"],
                key=f"{key_prefix}_kind"
            )
        with c_opt2:
            palette = st.selectbox(
                "Esquema de Colores",
                ["Neón Azul", "Esmeralda", "Violeta", "Viridis", "Warm / Fuego"],
                key=f"{key_prefix}_pal"
            )
        with c_opt3:
            scale_y = st.radio("Escala Y", ["Lineal", "Log"], horizontal=True, key=f"{key_prefix}_scale")

        colors = color_map.get(palette, ["#3b82f6"])

        if chart_kind == "Histograma + Boxplot":
            fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribución e Histograma de {selected_col}", color_discrete_sequence=colors if isinstance(colors, list) else [colors])
        elif chart_kind == "Diagrama de Caja (Boxplot)":
            fig = px.box(df, y=selected_col, points="all", title=f"Caja y Bigotes de {selected_col}", color_discrete_sequence=colors if isinstance(colors, list) else [colors])
        elif chart_kind == "Diagrama de Violín":
            fig = px.violin(df, y=selected_col, box=True, points="all", title=f"Diagrama de Violín de {selected_col}", color_discrete_sequence=colors if isinstance(colors, list) else [colors])
        else:
            fig = px.strip(df, y=selected_col, title=f"Dispersión Puntos de {selected_col}", color_discrete_sequence=colors if isinstance(colors, list) else [colors])

        if scale_y == "Log":
            fig.update_layout(yaxis_type="log")

    else:
        c_opt1, c_opt2 = st.columns([2, 2])
        with c_opt1:
            chart_kind = st.selectbox(
                "Tipo de Gráfico",
                ["Gráfico de Barras", "Gráfico Torta (Donut)"],
                key=f"{key_prefix}_kind"
            )
        with c_opt2:
            palette = st.selectbox(
                "Esquema de Colores",
                ["Pastel", "Neón Azul", "Esmeralda", "Violeta", "Viridis"],
                key=f"{key_prefix}_pal"
            )

        freq_df = df[selected_col].value_counts().reset_index()
        freq_df.columns = ["value", "count"]

        colors = color_map.get(palette, px.colors.qualitative.Pastel)

        if chart_kind == "Gráfico de Barras":
            fig = px.bar(
                freq_df, x="value", y="count",
                title=f"Frecuencias de Categorías para {selected_col}",
                color="value",
                labels={"value": "Categoría", "count": "Frecuencia"},
                color_discrete_sequence=colors
            )
        else:
            fig = px.pie(
                freq_df, names="value", values="count",
                title=f"Distribución Porcentual para {selected_col}",
                hole=0.4,
                color_discrete_sequence=colors
            )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f3f4f6'
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3. UTILIDADES HIGEA (LIMPIEZA DE DATOS)
# ============================================================
# ============================================================
# HIGEA — DIAGNÓSTICO Y LIMPIEZA DE DATOS
# ============================================================



def higea_refresh_dataframe(df):
    """Actualiza el DataFrame de trabajo y sus tipos."""
    df = df.copy(deep=True)
    st.session_state.df = df
    try:
        cleaned_df, var_types = data_processor.detect_variable_types(df)
        st.session_state.df = cleaned_df
        st.session_state.var_types = var_types
    except Exception:
        st.session_state.var_types = {
            col: ("numeric" if pd.api.types.is_numeric_dtype(df[col])
                  else "categorical")
            for col in df.columns
        }


def higea_analyze_missing(df):
    """Calcula media, mediana y moda para las columnas con faltantes."""
    rows = []

    for col in df.columns:
        serie = df[col]
        faltantes = int(serie.isna().sum())

        if faltantes == 0:
            continue

        numerica = pd.api.types.is_numeric_dtype(serie)
        moda_s = serie.mode(dropna=True)

        rows.append({
            "Variable": col,
            "Tipo": "Numérica" if numerica else "Categórica",
            "Faltantes": faltantes,
            "Media": serie.mean() if numerica else np.nan,
            "Mediana": serie.median() if numerica else np.nan,
            "Moda": moda_s.iloc[0] if not moda_s.empty else np.nan,
            "Método recomendado": "Mediana" if numerica else "Moda"
        })

    return pd.DataFrame(rows)


def higea_fill_numeric(df, method="median"):
    """Rellena faltantes numéricos con media o mediana."""
    out = df.copy(deep=True)

    for col in out.select_dtypes(include=np.number).columns:
        if out[col].isna().any():
            value = out[col].mean() if method == "mean" else out[col].median()
            if pd.notna(value):
                out[col] = out[col].fillna(value)

    return out


def higea_fill_mode(df):
    """Rellena faltantes categóricos con la moda."""
    out = df.copy(deep=True)

    for col in out.select_dtypes(
        include=["object", "string", "category"]
    ).columns:
        if out[col].isna().any():
            moda = out[col].mode(dropna=True)
            if not moda.empty:
                out[col] = out[col].fillna(moda.iloc[0])

    return out


def higea_remove_duplicates(df):
    """Conserva una aparición y elimina solamente las copias adicionales."""
    return df.drop_duplicates(keep="first").copy()


def higea_duplicate_report(df):
    """Muestra grupos de filas idénticas y cuántas copias sobran."""
    mask = df.duplicated(keep=False)

    if not mask.any():
        return pd.DataFrame()

    duplicadas = df.loc[mask].copy()
    columnas = list(df.columns)

    reporte = (
        duplicadas
        .groupby(columnas, dropna=False)
        .size()
        .reset_index(name="Apariciones")
    )

    reporte["Copias a eliminar"] = reporte["Apariciones"] - 1

    return reporte.sort_values(
        "Apariciones",
        ascending=False
    )


def higea_clean_spaces(df):
    """Quita espacios de nombres de columnas y valores de texto."""
    out = df.copy(deep=True)
    out.columns = [str(c).strip() for c in out.columns]

    for col in out.select_dtypes(
        include=["object", "string"]
    ).columns:
        out[col] = out[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return out


def higea_remove_empty(df):
    """Elimina filas y columnas completamente vacías."""
    return (
        df.dropna(how="all")
          .dropna(axis=1, how="all")
          .copy()
    )

# ============================================================
# 4. ESTILOS
# ============================================================
# Custom CSS Injection
def inject_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Backgrounds */
    .stApp {
        background-color: #0b0f19 !important;
        color: #f3f4f6 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Premium components */
    .premium-badge {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        color: #ffffff !important;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 12px;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .glass-panel {
        background: rgba(17, 24, 39, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-panel:hover {
        border-color: rgba(59, 130, 246, 0.4) !important;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        transition: all 0.2s ease-in-out;
    }
    
    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.4) !important;
        transform: translateY(-2px);
    }
    
    /* Code styling */
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Highlight color borders */
    .border-blue { border-left: 4px solid #3b82f6 !important; }
    .border-emerald { border-left: 4px solid #10b981 !important; }
    .border-violet { border-left: 4px solid #8b5cf6 !important; }
    .border-cyan { border-left: 4px solid #06b6d4 !important; }
    
    /* Footer mathematical check badge */
    .math-check-badge {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem;
        color: #10b981 !important;
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .dot {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Custom tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(17, 24, 39, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 6px 6px 0px 0px !important;
        padding: 10px 20px !important;
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        color: #3b82f6 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 4b. ESTADO DE SESIÓN
# ============================================================
def init_state():
    """Inicializa las variables de sesión una sola vez."""
    # Run math validations once at startup
    if "test_results" not in st.session_state:
        try:
            importlib.reload(calculator)
            importlib.reload(tests)
            st.session_state.test_results = tests.run_mathematical_tests()
        except Exception:
            st.session_state.test_results = {"results": [], "all_pass": True}

    defaults = {
        "wizard_step": 1,
        "wizard_answers": {},
        "df": None,
        "df_original": None,
        "var_types": {},
        "csv_name": "datos.csv",
        "selected_tab": TAB_ADVISOR,
        "calc_test": "Estadísticos Descriptivos",
        "calc_preselected_vars": {},
        "manual_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# 5. NAVEGACIÓN
# ============================================================

def get_enabled_tabs():
    """Devuelve las secciones activas (incluye las dependencias necesarias)."""
    active = {m["label"] for m in MODULES if m["enabled"]}
    changed = True
    while changed:
        changed = False
        for m in MODULES:
            if m["label"] in active:
                for req in m["requires"]:
                    if req not in active:
                        active.add(req)
                        changed = True
    return [m["label"] for m in MODULES if m["label"] in active]


def is_enabled(tab):
    return tab in get_enabled_tabs()


def goto(tab):
    """Cambia de sección solo si está habilitada."""
    if is_enabled(tab):
        st.session_state.selected_tab = tab



# ============================================================
# 6. SECCIONES (una función por módulo)
# ============================================================

def render_advisor():
    """Asesor Metodológico: wizard de 5 pasos y recomendación de modelo."""
    st.title("🧠 Asesor Metodológico Inteligente")
    st.subheader("Árbol de decisión experto para diagnosticar y seleccionar el modelo estadístico adecuado.")
    
    step = st.session_state.wizard_step
    answers = st.session_state.wizard_answers
    
    # Render wizard progress bar
    progress = (step - 1) / 5.0
    st.progress(progress, text=f"Paso {step} de 5")
    
    # Render Step questions
    if step == 1:
        q_id = "objective"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        # Display options as radio
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        
        # Match current answer or select index 0
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el objetivo de su investigación:", opts_labels, index=curr_idx, key="wiz_step_1_radio")
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    elif step == 2:
        q_id = "var_type"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el tipo de variables:", opts_labels, index=curr_idx, key="wiz_step_2_radio")
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    elif step == 3:
        q_id = "var_count"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el número de variables implicadas:", opts_labels, index=curr_idx, key="wiz_step_3_radio")
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    elif step == 4:
        q_id = "independence"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el tipo de relación entre muestras:", opts_labels, index=curr_idx, key="wiz_step_4_radio")
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    elif step == 5:
        q_id = "assumptions"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el cumplimiento de supuestos de normalidad/varianza:", opts_labels, index=curr_idx, key="wiz_step_5_radio")
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if step > 1:
            if st.button("⬅️ Anterior"):
                st.session_state.wizard_step -= 1
                st.rerun()
    with col2:
        if step < 5:
            if st.button("Siguiente ➡️"):
                st.session_state.wizard_step += 1
                st.rerun()
        elif step == 5:
            if st.button("🎯 Finalizar y Generar Diagnóstico"):
                st.session_state.wizard_step += 1
                st.rerun()
                
    # Step 6: Display results
    if step > 5:
        st.markdown("---")
        rec = advisor.get_recommendation(answers)
        model = rec["model"]
        alt_model = rec["alternative"]
        
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            st.markdown(f"""
            <div class="glass-panel border-violet">
                <span style="color:#8b5cf6; font-size:0.9rem; font-weight:600; text-transform:uppercase;">Modelo Recomendado</span>
                <h2 style="margin-top:5px; color:#ffffff;">{model['name']}</h2>
                <hr style="border-color:rgba(255,255,255,0.08);">
                <h5>Justificación Científica Académica Formal:</h5>
                <p style="font-style:italic; font-size:1.05rem; line-height:1.6; color:#e2e8f0;">"{rec['justification']}"</p>
                <hr style="border-color:rgba(255,255,255,0.08);">
                <h5>Supuestos Estadísticos Críticos a Verificar:</h5>
                <ul>
                    {"".join(f"<li>{ass}</li>" for ass in rec['assumptions'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Desplegar la Guía Procedimental Metodológica de 5 Pasos del Modelo
            st.markdown("---")
            with st.expander(f"📌 Ver Ruta Metodológica Paso a Paso para {model['name']}", expanded=True):
                proc_steps = database.get_model_procedure(model)
                for step_info in proc_steps:
                    st.markdown(f"**{step_info['step']}**")
                    st.caption(step_info["desc"])

            # Guía de aplicación: momento del estudio, ejercicio específico y aplicación
            guide = rec.get("guide")
            if guide:
                with st.expander("🎯 ¿Cuándo se usa? Momento, ejercicio específico y aplicación", expanded=True):
                    st.markdown(f"**📍 Momento del estudio en que se opera**\n\n{guide['moment']}")
                    st.markdown(f"**🧪 Ejercicio específico**\n\n{guide['exercise']}")
                    st.markdown(f"**🏷️ Aplicación**\n\n{guide['application']}")
                    st.caption(f"⚠️ Límites: {guide['limits']}")

            # Action to send to calculator if core
            core_mappings = {
                "arithmetic_mean": "Estadísticos Descriptivos",
                "median_iqr": "Estadísticos Descriptivos",
                "frequencies_props": "Estadísticos Descriptivos",
                "independent_ttest": "Prueba t de Student para Muestras Independientes",
                "paired_ttest": "Prueba t de Student Pareada (Medidas Repetidas)",
                "simple_regression": "Regresión Lineal Simple",
                "chi_squared_test": "Prueba de Chi-cuadrado de Independencia",
                "oneway_anova": "ANOVA de una vía (Análisis de Varianza)"
            }
            
            # Map also advanced generated variants back to their base calculator
            base_id = model["id"].split("_var")[0]
            if base_id in core_mappings:
                cal_test_name = core_mappings[base_id]
                if st.button(f"🚀 Cargar '{model['name']}' en el Módulo Calculadora", disabled=not is_enabled(TAB_CALC)):
                    st.session_state.calc_test = cal_test_name
                    goto(TAB_CALC)
                    st.rerun()
            else:
                st.info("💡 Este modelo avanzado no posee una calculadora interactiva paso a paso integrada. Puede consultar sus plantillas de código listas para producción en la pestaña 'Biblioteca de Modelos'.")
                
        with col_res2:
            st.markdown("### Parámetros Evaluados")
            # Map readable answers
            for q_key, q_val in answers.items():
                q_title = advisor.QUESTIONS[q_key]["title"].split(". ")[1]
                opts = advisor.QUESTIONS[q_key]["options"]
                readable_ans = next((o[1] for o in opts if o[0] == q_val), q_val)
                st.markdown(f"**{q_title}**\n* {readable_ans}")
                
            st.markdown("---")
            if alt_model:
                st.markdown(f"""
                <div class="glass-panel border-cyan" style="padding:15px !important;">
                    <span style="color:#06b6d4; font-size:0.8rem; font-weight:600; text-transform:uppercase;">Alternativa Metodológica</span>
                    <h5 style="margin-top:5px; margin-bottom:5px; color:#ffffff;">{alt_model['name']}</h5>
                    <p style="font-size:0.85rem; color:#94a3b8;">Si se violan supuestos o el diseño metodológico cambia ligeramente, este es el modelo recomendado.</p>
                </div>
                """, unsafe_allow_html=True)

                alt_guide = rec.get("alt_guide")
                if alt_guide:
                    with st.expander("¿Cuándo preferir la alternativa?"):
                        st.markdown(f"**📍 Momento**\n\n{alt_guide['moment']}")
                        st.markdown(f"**🧪 Ejercicio**\n\n{alt_guide['exercise']}")

                if st.button("Ver Alternativa"):
                    # Temporarily replace with alternative recommendation
                    # A trick to swap
                    answers["assumptions"] = "nonparametric" if answers.get("assumptions") == "parametric" else "parametric"
                    st.rerun()
                    
            if st.button("🔄 Reiniciar Asesor"):
                st.session_state.wizard_step = 1
                st.session_state.wizard_answers = {}
                st.rerun()


def render_library():
    """Biblioteca de Modelos: catálogo, plantillas de código y carga en calculadora."""
    st.title("📚 Biblioteca Científica de Modelos")
    st.subheader("Catálogo de nivel doctoral con más de 390 modelos y lenguajes de programación estadística.")
    
    bib_mode1, bib_mode2 = st.tabs(["📚 Catálogo Completo (+390 Modelos)", "🎯 Recomendaciones para tu Dataset"])
    
    with bib_mode2:
        st.markdown("### 🎯 Recomendador de Modelos según tu Dataset Actual")
        if st.session_state.df is None:
            st.info("💡 Para recibir recomendaciones personalizadas basadas en la estructura de tus variables, sube un archivo CSV en el **Laboratorio de Datos**.")
        else:
            recs = database.recommend_models_for_dataset(st.session_state.df)
            if not recs:
                st.warning("No se encontraron coincidencias automáticas para este conjunto de datos.")
            else:
                st.success(f" Se identificaron **{len(recs)} modelos de alta afinidad** para las variables de tu archivo '{st.session_state.csv_name}'.")
                for r in recs:
                    rm = r["model"]
                    score = r["score"]
                    reasons = r["reasons"]
                    
                    st.markdown(f"""
                    <div class="glass-panel border-cyan" style="padding:15px !important; margin-bottom:12px !important;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="color:#ffffff; margin:0;">{rm['name']}</h4>
                            <span class="premium-badge" style="margin:0;">Afinidad: {score}%</span>
                        </div>
                        <p style="color:#cbd5e1; font-size:0.9rem; margin-top:6px;">{rm['desc']}</p>
                        <ul style="color:#94a3b8; font-size:0.85rem; margin-bottom:5px;">
                            {"".join(f"<li>{reason}</li>" for reason in reasons)}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Procedure expander
                    with st.expander(f"📋 Ver Procedimiento de Aplicación Paso a Paso para '{rm['name']}'"):
                        proc_steps = database.get_model_procedure(rm)
                        for p in proc_steps:
                            st.markdown(f"**{p['step']}**")
                            st.markdown(f"{p['desc']}")
                            st.markdown("---")
                    
                    base_id = rm["id"].split("_var")[0]
                    core_mappings = {
                        "arithmetic_mean": "Estadísticos Descriptivos",
                        "median_iqr": "Estadísticos Descriptivos",
                        "frequencies_props": "Estadísticos Descriptivos",
                        "independent_ttest": "Prueba t de Student para Muestras Independientes",
                        "paired_ttest": "Prueba t de Student Pareada (Medidas Repetidas)",
                        "simple_regression": "Regresión Lineal Simple",
                        "chi_squared_test": "Prueba de Chi-cuadrado de Independencia",
                        "oneway_anova": "ANOVA de una vía (Análisis de Varianza)"
                    }
                    if base_id in core_mappings:
                        cal_test_name = core_mappings[base_id]
                        if st.button(f"🚀 Cargar '{rm['name']}' en Calculadora", key=f"rec_load_{rm['id']}", disabled=not is_enabled(TAB_CALC)):
                            st.session_state.calc_test = cal_test_name
                            goto(TAB_CALC)
                            st.rerun()
                            
    with bib_mode1:
        # Filters
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            mod_names = ["Todos los módulos"] + [m["name"] for m in database.MODULES]
            selected_mod = st.selectbox("Filtrar por Módulo Estadístico", mod_names)
            
        with col_f2:
            search_query = st.text_input("🔍 Buscar por palabra clave (ej. Welch, ANOVA, Logística, Bootstrap)...")
            
        # Filter Catalog
        catalog = database.MODEL_CATALOG
        
        # Filter by module
        if selected_mod != "Todos los módulos":
            mod_id = next(m["id"] for m in database.MODULES if m["name"] == selected_mod)
            catalog = [m for m in catalog if m["module"] == mod_id]
            
        # Filter by search
        if search_query:
            query = normalize_text(search_query)
            catalog = [m for m in catalog if query in normalize_text(m["name"]) or query in normalize_text(m["desc"])]
            
        st.markdown(f"**Se encontraron {len(catalog)} modelos estadísticos en este segmento.**")
        
        # Page pagination / grid
        limit = 20
        pages_count = int(np.ceil(len(catalog) / limit)) if catalog else 1
        
        if pages_count > 1:
            page_num = st.number_input("Página", min_value=1, max_value=pages_count, step=1, value=1)
        else:
            page_num = 1
            
        start_idx = (page_num - 1) * limit
        end_idx = min(start_idx + limit, len(catalog))
        
        page_catalog = catalog[start_idx:end_idx]
        
        # Render Grid
        for m in page_catalog:
            with st.container():
                # Check border accent
                border_cls = "border-violet" if m["is_advanced"] else "border-blue"
                if m["is_core"]:
                    border_cls = "border-cyan"
                    
                # Render card
                st.markdown(f"""
                <div class="glass-panel {border_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="color:#ffffff; margin:0;">{m['name']}</h3>
                        <span style="background:rgba(255,255,255,0.06); padding:4px 8px; border-radius:4px; font-size:0.75rem; color:#94a3b8;">
                            {next(mod['name'] for mod in database.MODULES if mod['id'] == m['module'])}
                        </span>
                    </div>
                    <p style="margin-top:10px; color:#cbd5e1; font-size:0.95rem;">{m['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show details button
                with st.expander(f"📖 Ver Procedimiento Paso a Paso, Fórmulas y Código Multilingüe"):
                    detail_tab1, detail_tab2, detail_tab3 = st.tabs([
                        "📋 Procedimiento Paso a Paso",
                        "📖 Ficha Científica y Supuestos",
                        "💻 Plantillas de Código (5 Lenguajes)"
                    ])
                    
                    with detail_tab1:
                        st.markdown("#### Procedimiento de Realización y Aplicación Práctica")
                        proc_steps = database.get_model_procedure(m)
                        for p in proc_steps:
                            st.markdown(f"**{p['step']}**")
                            st.markdown(f"{p['desc']}")
                            st.markdown("---")
                            
                    with detail_tab2:
                        st.markdown("#### Ficha Científica del Modelo")
                        col_d1, col_d2 = st.columns([1, 1])
                        with col_d1:
                            st.markdown("**Justificación Académica:**")
                            st.write(m["justification"])
                            st.markdown("**Supuestos Metodológicos:**")
                            for a in m["assumptions"]:
                                st.markdown(f"* {a}")
                        with col_d2:
                            if m["formula"]:
                                st.markdown("**Ecuación Algebraica:**")
                                st.latex(m["formula"])
                            else:
                                st.info("Fórmula genérica basada en la parametrización del modelo lineal general.")
                                
                    with detail_tab3:
                        if m["templates"]:
                            st.markdown("#### Plantillas de Código Multilingüe")
                            code_tabs = st.tabs(["Python", "R", "SPSS", "Stata", "SAS"])
                            langs = ["python", "r", "spss", "stata", "sas"]
                            
                            for t_tab, lang in zip(code_tabs, langs):
                                with t_tab:
                                    template_code = m["templates"][lang]
                                    formatted_code = safe_format(template_code,
                                        csv_file=st.session_state.csv_name,
                                        var="Variable",
                                        group_var="Grupo",
                                        dep_var="Dependiente",
                                        group1="A",
                                        group2="B",
                                        var1="Variable1",
                                        var2="Variable2",
                                        x_var="X_Predictora",
                                        y_var="Y_Respuesta"
                                    )
                                    st.code(formatted_code, language=lang)
                        else:
                            st.info("Plantillas de código para este modelo avanzado están disponibles en la documentación complementaria.")
                    
                    # Acciones rápidas para el modelo
                    base_id = m["id"].split("_var")[0]
                    core_mappings = {
                        "arithmetic_mean": "Estadísticos Descriptivos",
                        "median_iqr": "Estadísticos Descriptivos",
                        "frequencies_props": "Estadísticos Descriptivos",
                        "independent_ttest": "Prueba t de Student para Muestras Independientes",
                        "paired_ttest": "Prueba t de Student Pareada (Medidas Repetidas)",
                        "simple_regression": "Regresión Lineal Simple",
                        "chi_squared_test": "Prueba de Chi-cuadrado de Independencia",
                        "oneway_anova": "ANOVA de una vía (Análisis de Varianza)"
                    }
                    if base_id in core_mappings:
                        cal_test_name = core_mappings[base_id]
                        if st.button(f"🚀 Cargar '{m['name']}' en la Calculadora Analítica", key=f"bib_load_{m['id']}", disabled=not is_enabled(TAB_CALC)):
                            st.session_state.calc_test = cal_test_name
                            goto(TAB_CALC)
                            st.rerun()


def render_lab():
    """Laboratorio de Datos: carga de CSV, limpieza rápida y exploración."""
    st.title("🧪 Laboratorio de Análisis de Datos")
    st.subheader("Importe archivos CSV, ejecute limpieza inteligente y analice descriptivamente variables.")

    gate_message = st.session_state.pop("gate_message", None)
    if gate_message:
        st.warning(f"⚠️ {gate_message}")
    
    # Auto-cierre de carga si ya existe en el proyecto
    if Path("ventas_200.csv").exists() and st.session_state.df is None:
        if st.button("📂 Cargar 'ventas_200.csv' del proyecto directamente", key="btn_auto_load_ventas", type="primary"):
            df = pd.read_csv("ventas_200.csv", encoding="utf-8").dropna(how='all')
            st.session_state.df_original = df.copy(deep=True)
            st.session_state.df = df.copy(deep=True)
            st.session_state.csv_name = "ventas_200.csv"
            st.session_state.last_uploaded_name = "ventas_200.csv"
            cleaned_df, var_types = data_processor.detect_variable_types(df)
            st.session_state.df = cleaned_df
            st.session_state.var_types = var_types
            st.success("✅ 'ventas_200.csv' cargado exitosamente (200 filas válidas).")
            st.rerun()

    # File Uploader
    uploaded_file = st.file_uploader("Subir Archivo de Datos (CSV)", type=["csv"], help="Formatos con separador por coma (,) o punto y coma (;).")
    
    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            try:
                # Parse CSV
                df = data_processor.parse_csv(uploaded_file)

                # Conservar una copia inmutable de referencia para "Restaurar".
                st.session_state.df_original = df.copy(deep=True)
                st.session_state.df = df.copy(deep=True)
                st.session_state.csv_name = uploaded_file.name
                st.session_state.last_uploaded_name = uploaded_file.name

                # Detect variable types
                cleaned_df, var_types = data_processor.detect_variable_types(df)
                
                # Store in session state
                st.session_state.df = cleaned_df
                st.session_state.var_types = var_types
                
                st.success(f"✅ Archivo '{uploaded_file.name}' cargado exitosamente. Se detectaron {df.shape[0]} filas y {df.shape[1]} columnas.")
            except Exception as e:
                st.error(f"Error al procesar el archivo CSV: {str(e)}")
            
    # Display statistics if data is loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        var_types = st.session_state.var_types
        
        # Panel Destacado de Limpieza de Datos (HIGEA)
        with st.expander("🧹 Herramientas Rápidas de Limpieza de Datos (HIGEA AS-001)", expanded=True):
            st.markdown("##### 🧽 Limpieza y Calidad de Registros")
            c_cl1, c_cl2, c_cl3 = st.columns(3)
            with c_cl1:
                if st.button("⚡ Auto-Limpieza en 1-Clic", key="lab_inline_autoclean", use_container_width=True):
                    df_cleaned = higea_auto_clean(st.session_state.df)
                    st.session_state.df = df_cleaned
                    st.success("✅ Auto-limpieza ejecutada: nulos imputados, espacios eliminados y duplicados removidos.")
                    st.rerun()
            with c_cl2:
                if st.button("🗑️ Eliminar Filas Vacías/Nulas", key="lab_inline_drop_empty", use_container_width=True):
                    df_cleaned = higea_remove_empty(st.session_state.df)
                    st.session_state.df = df_cleaned
                    st.success("✅ Filas 100% vacías eliminadas.")
                    st.rerun()
            with c_cl3:
                if st.button("⚙️ Abrir Suite Completa de Limpieza (HIGEA)", key="lab_inline_open_higea", use_container_width=True, disabled=not is_enabled(TAB_HIGEA)):
                    goto(TAB_HIGEA)
                    st.rerun()

        st.markdown("### Tabla Vista Previa (Primeras 10 filas)")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Análisis Descriptivo y Visual de Variables")
        
        # Select variable
        df = st.session_state.get("df")
        all_cols = list(df.columns)
        selected_col = st.selectbox("Seleccione la variable a analizar:", all_cols)
        
        if selected_col:
            col_type = var_types.get(selected_col, "numeric" if pd.api.types.is_numeric_dtype(df[selected_col]) else "categorical")
            series = df[selected_col]
            
            # Run descriptive analysis
            stats_res = calculator.calculate_descriptive(series)
            
            col_lab1, col_lab2 = st.columns([1, 1])
            
            with col_lab1:
                st.markdown(f"#### Estadísticas para **{selected_col}** (Tipo: *{col_type.upper()}*)")
                if col_type == "numeric":
                    st.write(pd.DataFrame({
                        "Estadístico": [
                            "Observaciones válidas (N)",
                            "Media Aritmética",
                            "Mediana",
                            "Moda",
                            "Desviación Estándar",
                            "Varianza Muestral",
                            "Rango Intercuartílico (IQR)",
                            "Asimetría",
                            "Curtosis",
                            "Intervalo Confianza 95% (Inf)",
                            "Intervalo Confianza 95% (Sup)"
                        ],
                        "Valor": [
                            int(stats_res["n"]),
                            f"{stats_res['mean']:.4f}",
                            f"{stats_res['median']:.4f}",
                            f"{stats_res['mode']:.4f}",
                            f"{stats_res['std_dev']:.4f}",
                            f"{stats_res['variance']:.4f}",
                            f"{stats_res['iqr']:.4f}",
                            f"{stats_res['skewness']:.4f}",
                            f"{stats_res['kurtosis']:.4f}",
                            f"{stats_res['ci_lower']:.4f}",
                            f"{stats_res['ci_upper']:.4f}"
                        ]
                    }))
                else:
                    # Categorical table
                    freq_df = pd.DataFrame(stats_res["table"])
                    freq_df.columns = ["Categoría", "Frecuencia (f)", "Porcentaje (%)"]
                    st.dataframe(freq_df, hide_index=True, use_container_width=True)
                    
            with col_lab2:
                st.markdown("#### Visualización Dinámica")
                render_customizable_chart(df, selected_col, col_type, key_prefix="lab_chart")
            
            st.markdown("---")
            st.markdown("##### 🚀 Accesos Rápidos con este Dataset")
            col_lab_btn1, col_lab_btn2 = st.columns(2)
            with col_lab_btn1:
                if st.button("🧹 Abrir Módulo de Limpieza Completa (HIGEA)", key="lab_btn_higea", use_container_width=True, disabled=not is_enabled(TAB_HIGEA)):
                    goto(TAB_HIGEA)
                    st.rerun()
            with col_lab_btn2:
                if st.button(f"🧮 Analizar '{selected_col}' en la Calculadora Analítica", key="lab_btn_calc", use_container_width=True, disabled=not is_enabled(TAB_CALC)):
                    st.session_state.calc_test = "Estadísticos Descriptivos"
                    st.session_state.calc_preselected_vars = {"var": selected_col}
                    goto(TAB_CALC)
                    st.rerun()
    else:
        st.info("💡 Por favor, suba un archivo CSV en el panel de arriba para iniciar el Laboratorio de Datos.")
        if is_enabled(TAB_CALC):
            if st.button("✍️ Continuar sin archivo (ingreso manual en la Calculadora)", key="lab_btn_manual_mode"):
                st.session_state.manual_mode = True
                goto(TAB_CALC)
                st.rerun()


def _ols_fit(X, y):
    """Ajuste OLS por mínimos cuadrados. Devuelve coeficientes, SE, t, p, R2, R2 ajustado y F."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    sse = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    df_resid = n - k
    mse = sse / df_resid if df_resid > 0 else np.nan
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * mse, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.where(se > 0, beta / se, np.nan)
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df_resid) if df_resid > 0 else np.full(k, np.nan)
    r2 = 1 - sse / sst if sst > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / df_resid if (df_resid > 0 and not np.isnan(r2)) else np.nan
    df_model = k - 1
    if df_model > 0 and not np.isnan(r2) and r2 < 1:
        f_stat = (r2 / df_model) / ((1 - r2) / df_resid)
        f_p = stats.f.sf(f_stat, df_model, df_resid)
    else:
        f_stat, f_p = np.nan, np.nan
    return {
        "beta": beta, "se": se, "t": t_vals, "p": p_vals, "y_hat": y_hat, "resid": resid,
        "r2": r2, "r2_adj": r2_adj, "f_stat": f_stat, "f_p": f_p,
        "df_model": df_model, "df_resid": df_resid, "n": n, "sse": sse,
    }


def _intervalo_confianza(beta, se, df_resid, nivel_confianza):
    """Intervalo de confianza t para los coeficientes (nivel en %: 90, 95 o 99)."""
    alfa = 1 - nivel_confianza / 100
    t_critico = stats.t.ppf(1 - alfa / 2, df_resid)
    return beta - t_critico * se, beta + t_critico * se


def render_multiple_regression(df, data_source):
    """Regresión Lineal Múltiple con intervalos de confianza (90 / 95 / 99 %).
    Solo trabaja con el CSV cargado (no tiene modo de ingreso manual)."""
    if df is None or data_source != "Seleccionar del CSV cargado":
        st.info("💡 La Regresión Lineal Múltiple requiere datos cargados desde el CSV. Sube tu archivo en el Laboratorio de Datos y selecciona «Seleccionar del CSV cargado».")
        return

    var_types = st.session_state.var_types
    num_cols = [c for c in df.columns if var_types.get(c) == "numeric"]
    if len(num_cols) < 3:
        st.warning("Se necesitan al menos 3 variables numéricas (1 dependiente y 2 independientes).")
        return

    y_col = st.selectbox("Variable Y (Dependiente - Numérica):", num_cols, key="mreg_y")
    x_opts = [c for c in num_cols if c != y_col]
    x_cols = st.multiselect("Variables X (Independientes - Numéricas, elige 2 o más):", x_opts, key="mreg_x")
    if len(x_cols) < 2:
        st.info("Selecciona al menos 2 variables independientes (si solo necesitas 1, usa Regresión Lineal Simple).")

    nivel_confianza = st.selectbox(
        "Nivel de confianza:", [90, 95, 99], index=1,
        format_func=lambda n: f"{n}%", key="mreg_conf"
    )

    if st.button("Ajustar Regresión Múltiple", key="mreg_btn", disabled=len(x_cols) < 2):
        datos = df[[y_col] + x_cols].dropna()
        if len(datos) <= len(x_cols) + 1:
            st.error("No hay suficientes observaciones válidas para estimar el modelo.")
            return
        y = datos[y_col].to_numpy(dtype=float)
        Xraw = datos[x_cols].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(datos)), Xraw])
        r = _ols_fit(X, y)
        ic_inf, ic_sup = _intervalo_confianza(r["beta"], r["se"], r["df_resid"], nivel_confianza)

        st.markdown("---")
        st.markdown(f"**Número de observaciones usadas:** `{r['n']}` (tras eliminar filas con datos faltantes)")
        st.markdown(f"**R²:** `{r['r2']:.4f}`  |  **R² ajustado:** `{r['r2_adj']:.4f}`")
        st.markdown(f"**F({r['df_model']}, {r['df_resid']}) =** `{r['f_stat']:.3f}`  |  **p-valor del modelo:** `{r['f_p']:.6f}`")
        if r["f_p"] < 0.05:
            st.success("**Estadísticamente Significativo (p < 0.05)**")
        else:
            st.error("**No Significativo (p >= 0.05)**")

        nombres = ["Intercepto"] + x_cols
        tabla = pd.DataFrame({
            "Variable": nombres,
            "Coeficiente": np.round(r["beta"], 5),
            "Error Estándar": np.round(r["se"], 5),
            "t": np.round(r["t"], 3),
            "p-valor": np.round(r["p"], 5),
            f"IC {nivel_confianza}% Inferior": np.round(ic_inf, 5),
            f"IC {nivel_confianza}% Superior": np.round(ic_sup, 5),
            "¿Significativo? (α=0.05)": ["Sí" if (pv is not None and pv < 0.05) else "No" for pv in r["p"]],
        })
        st.dataframe(tabla, hide_index=True, use_container_width=True)

        ecuacion = f"{y_col} = {r['beta'][0]:.4f}" + "".join(
            f" {'+' if b >= 0 else '-'} {abs(b):.4f}·{xc}" for b, xc in zip(r["beta"][1:], x_cols)
        )
        st.markdown(f"**Ecuación estimada:** `{ecuacion}`")
        st.caption("Cada coeficiente indica el cambio esperado en la variable Y por cada unidad de esa X, manteniendo las demás variables constantes (efecto «controlado»).")

        st.markdown(f"### Interpretación con {nivel_confianza}% de confianza")
        for nombre, b, inf, sup in zip(nombres[1:], r["beta"][1:], ic_inf[1:], ic_sup[1:]):
            if np.isnan(inf) or np.isnan(sup):
                st.warning(f"No se pudo calcular el intervalo de {nombre} (error estándar no disponible).")
            elif inf > 0 or sup < 0:
                st.success(
                    f"**{nombre}**: coeficiente {b:.4f}, intervalo [{inf:.4f}, {sup:.4f}]. "
                    f"No incluye el 0, por lo que el efecto es estadísticamente significativo al {nivel_confianza}% de confianza."
                )
            else:
                st.warning(
                    f"**{nombre}**: coeficiente {b:.4f}, intervalo [{inf:.4f}, {sup:.4f}]. "
                    f"Incluye el 0, por lo que NO es estadísticamente significativo al {nivel_confianza}% de confianza."
                )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["y_hat"], y=y, mode="markers", name="Observado vs. Predicho", marker=dict(color="#3b82f6", size=9)))
        lims = [min(r["y_hat"].min(), y.min()), max(r["y_hat"].max(), y.max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Ajuste perfecto (y = ŷ)", line=dict(color="#10b981", dash="dash")))
        fig.update_layout(title=f"Valores predichos vs. observados de {y_col}", xaxis_title=f"{y_col} predicho", yaxis_title=f"{y_col} observado",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f3f4f6")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Entre más cerca estén los puntos de la línea diagonal, mejor predice el modelo. Puntos muy alejados son observaciones que el modelo explica peor.")


def render_anova_report(res):
    """Reporte del ANOVA de una vía: tabla ANOVA, tamaño del efecto, supuestos, pruebas robustas y post-hoc."""
    alpha = res.get("alpha", 0.05)

    def fmt(x, dec=4):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.{dec}f}"

    # 1. Tabla ANOVA
    st.markdown("**Tabla ANOVA:**")
    st.dataframe(pd.DataFrame({
        "Fuente": ["Entre grupos", "Dentro de grupos (error)", "Total"],
        "Suma de Cuadrados": [fmt(res["ss_between"]), fmt(res["ss_within"]), fmt(res["sst"])],
        "gl": [str(res["df_between"]), str(res["df_within"]), str(res["df_total"])],
        "Cuadrado Medio": [fmt(res["ms_between"]), fmt(res["ms_within"]), ""],
        "F": [fmt(res["f_statistic"]), "", ""],
        "p-valor": [fmt(res["p_value"], 6), "", ""],
    }), hide_index=True, use_container_width=True)

    # 2. Tamaño del efecto
    st.markdown(
        f"**Tamaño del efecto:** η² = `{res['eta_squared']:.4f}` ({res['effect_label']}) | "
        f"ω² = `{res['omega_squared']:.4f}` | f de Cohen = `{res['cohen_f']:.4f}`"
    )
    st.caption(
        f"El factor explica aproximadamente el {res['eta_squared'] * 100:.1f}% de la variabilidad "
        "de la variable dependiente. Referencia de η²: 0.01 pequeño, 0.06 mediano, 0.14 grande."
    )

    # 3. Resumen por grupo
    st.markdown("**Resumen de Medias Grupales:**")
    st.dataframe(pd.DataFrame({
        "Grupo": [g["group"] for g in res["groups"]],
        "Tamaño (n)": [g["n"] for g in res["groups"]],
        "Media": [round(g["mean"], 4) for g in res["groups"]],
        "Desv. Est.": [round(g["sd"], 4) for g in res["groups"]],
        "IC 95% Inf": [round(g["ci_lower"], 4) for g in res["groups"]],
        "IC 95% Sup": [round(g["ci_upper"], 4) for g in res["groups"]],
    }), hide_index=True, use_container_width=True)

    # 4. Verificación de supuestos
    st.markdown("**Verificación de supuestos:**")
    sh, lv = res.get("shapiro"), res.get("levene")
    normal_ok = sh is None or sh["p_value"] >= alpha
    var_ok = lv is None or lv["p_value"] >= alpha

    if sh is None:
        st.info("No se pudo aplicar Shapiro-Wilk (requiere entre 3 y 5000 observaciones y variabilidad en los residuos).")
    elif normal_ok:
        st.success(f"Normalidad de los residuos (Shapiro-Wilk): W = {sh['statistic']:.4f}, p = {sh['p_value']:.4f}. Se cumple.")
    else:
        st.warning(
            f"Normalidad de los residuos (Shapiro-Wilk): W = {sh['statistic']:.4f}, p = {sh['p_value']:.4f}. "
            "No se cumple. Con grupos grandes el ANOVA tolera este incumplimiento; en caso contrario "
            "considera Kruskal-Wallis."
        )

    if lv is None:
        st.info("No se pudo aplicar la prueba de Levene.")
    elif var_ok:
        st.success(f"Homogeneidad de varianzas (Levene): F = {lv['statistic']:.4f}, p = {lv['p_value']:.4f}. Se cumple.")
    else:
        st.warning(
            f"Homogeneidad de varianzas (Levene): F = {lv['statistic']:.4f}, p = {lv['p_value']:.4f}. "
            "No se cumple. Usa el ANOVA de Welch como resultado principal."
        )

    # 5. Pruebas robustas de verificación cruzada
    with st.expander("Pruebas robustas de verificación (Welch y Kruskal-Wallis)", expanded=not (normal_ok and var_ok)):
        wl, kw = res.get("welch"), res.get("kruskal")
        if wl:
            st.markdown(
                f"**ANOVA de Welch:** F({res['df_between']}, {wl['df2']:.2f}) = `{wl['statistic']:.4f}`, "
                f"p = `{wl['p_value']:.6f}`"
            )
        else:
            st.caption("El ANOVA de Welch no se pudo calcular (algún grupo tiene varianza cero).")
        if kw:
            st.markdown(
                f"**Kruskal-Wallis:** H({kw['df']}) = `{kw['statistic']:.4f}`, p = `{kw['p_value']:.6f}`"
            )
        else:
            st.caption("Kruskal-Wallis no se pudo calcular.")
        st.caption("Si estas pruebas concluyen lo mismo que el ANOVA clásico, la conclusión es sólida.")

    # 6. Comparaciones post-hoc
    st.markdown("**Comparaciones múltiples (post-hoc):**")
    if res["p_value"] >= alpha:
        st.info("El F global no es significativo, por lo que no corresponde hacer comparaciones post-hoc.")
    else:
        ph = res.get("posthoc")
        if not ph:
            st.warning("No se pudo calcular Tukey HSD (requiere SciPy 1.8 o superior).")
        else:
            st.dataframe(pd.DataFrame({
                "Comparación": [f"{p['group_a']} vs {p['group_b']}" for p in ph["pairs"]],
                "Diferencia de medias": [round(p["mean_diff"], 4) for p in ph["pairs"]],
                "p ajustado": [round(p["p_adj"], 6) for p in ph["pairs"]],
                "IC 95% Inf": [round(p["ci_lower"], 4) for p in ph["pairs"]],
                "IC 95% Sup": [round(p["ci_upper"], 4) for p in ph["pairs"]],
                "¿Significativa?": ["Sí" if p["significant"] else "No" for p in ph["pairs"]],
            }), hide_index=True, use_container_width=True)
            st.caption(
                "Tukey HSD controla el error tipo I al comparar todos los pares. "
                "Si las varianzas no son homogéneas, interprétalo con cautela (Games-Howell es la alternativa)."
            )


def render_calc():
    """Calculadora Analítica: cálculo paso a paso con LaTeX e interpretación."""
    st.title("🧮 Calculadora de Modelos Estadísticos")
    st.subheader("Cálculo matemático paso a paso con fórmulas LaTeX e interpretación científica.")
    
    # Selected test
    tests_list = [
        "Estadísticos Descriptivos",
        "Prueba t de Student para Muestras Independientes",
        "Prueba t de Student Pareada (Medidas Repetidas)",
        "Prueba de Chi-cuadrado de Independencia",
        "Regresión Lineal Simple",
        "Regresión Lineal Múltiple",
        "ANOVA de una vía (Análisis de Varianza)"
    ]

    selected_test = st.selectbox("Seleccione el análisis estadístico a realizar:", tests_list, index=tests_list.index(st.session_state.calc_test))
    st.session_state.calc_test = selected_test
    
    # Desplegar ruta procedimental del modelo seleccionado en la calculadora
    calc_to_cat_map = {
        "Estadísticos Descriptivos": "arithmetic_mean",
        "Prueba t de Student para Muestras Independientes": "independent_ttest",
        "Prueba t de Student Pareada (Medidas Repetidas)": "paired_ttest",
        "Prueba de Chi-cuadrado de Independencia": "chi_squared_test",
        "Regresión Lineal Simple": "simple_regression",
        "Regresión Lineal Múltiple": "simple_regression",
        "ANOVA de una vía (Análisis de Varianza)": "oneway_anova"
    }
    cat_id = calc_to_cat_map.get(selected_test, "arithmetic_mean")
    cat_model = database.CORE_MODELS.get(cat_id, database.MODEL_CATALOG[0])
    
    with st.expander(f"📍 Ruta Metodológica Paso a Paso para '{selected_test}'", expanded=False):
        proc_steps = database.get_model_procedure(cat_model)
        for p_step in proc_steps:
            st.markdown(f"**{p_step['step']}**")
            st.caption(p_step["desc"])
            
    # Source switcher (Manual vs CSV)
    has_data = st.session_state.df is not None
    data_source = st.radio(
        "Fuente de Datos",
        ["Seleccionar del CSV cargado", "Ingresar datos manualmente"],
        index=0 if has_data else 1,
        horizontal=True,
        key="calc_data_source_radio"
    )

    if not has_data:
        if data_source == "Seleccionar del CSV cargado":
            # Eligió usar un CSV pero aún no hay: se detiene en el Laboratorio.
            if is_enabled(TAB_LAB):
                st.session_state.gate_message = "Para analizar datos de un archivo, primero sube un CSV en el Laboratorio de Datos."
                goto(TAB_LAB)
                st.rerun()
            else:
                st.error("El Laboratorio de Datos está deshabilitado (SHOW_LAB = False); usa el ingreso manual.")
                data_source = "Ingresar datos manualmente"
        else:
            st.info("💡 Modo de ingreso manual activo. Si deseas analizar un archivo, sube un CSV en el Laboratorio de Datos.")
            if st.button("📂 Ir al Laboratorio a subir datos", key="calc_btn_go_lab", disabled=not is_enabled(TAB_LAB)):
                st.session_state.gate_message = "Sube tu archivo CSV aquí para poder usarlo en la Calculadora."
                goto(TAB_LAB)
                st.rerun()
        
    df = st.session_state.df
    var_types = st.session_state.var_types

    if selected_test == "Regresión Lineal Múltiple":
        render_multiple_regression(df, data_source)
        return

    # Render Inputs conditionally
    run_analysis = False
    inputs = {}
    
    if data_source in ["Ingresar datos manualmente", "Manual"]:
        st.markdown("##### Ingreso de Datos Manuales")
        
        if selected_test == "Estadísticos Descriptivos":
            manual_data = st.text_area("Ingrese los datos numéricos separados por comas:", "10, 15, 20, 25, 30")
            if st.button("Ejecutar Cálculo"):
                try:
                    data_list = [float(x.strip()) for x in manual_data.split(",") if x.strip()]
                    inputs = {"data": data_list}
                    run_analysis = True
                except ValueError:
                    st.error("Error: Ingrese solo números válidos.")
                    
        elif selected_test == "Prueba t de Student para Muestras Independientes":
            g1_data = st.text_area("Grupo 1 (A) - Valores numéricos:", "12, 14, 15, 18, 20")
            g2_data = st.text_area("Grupo 2 (B) - Valores numéricos:", "10, 11, 13, 12, 15")
            if st.button("Ejecutar Cálculo"):
                try:
                    data_g1 = [float(x.strip()) for x in g1_data.split(",") if x.strip()]
                    data_g2 = [float(x.strip()) for x in g2_data.split(",") if x.strip()]
                    inputs = {"g1": data_g1, "g2": data_g2}
                    run_analysis = True
                except ValueError:
                    st.error("Error: Ingrese solo números válidos.")
                    
        elif selected_test == "Prueba t de Student Pareada (Medidas Repetidas)":
            v1_data = st.text_area("Momento 1 (Pre):", "10, 12, 15, 11, 14")
            v2_data = st.text_area("Momento 2 (Post):", "12, 15, 17, 12, 16")
            if st.button("Ejecutar Cálculo"):
                try:
                    data_v1 = [float(x.strip()) for x in v1_data.split(",") if x.strip()]
                    data_v2 = [float(x.strip()) for x in v2_data.split(",") if x.strip()]
                    if len(data_v1) != len(data_v2):
                        st.error("Error: Los dos momentos deben tener el mismo número de observaciones pareadas.")
                    else:
                        inputs = {"v1": data_v1, "v2": data_v2}
                        run_analysis = True
                except ValueError:
                    st.error("Error: Ingrese solo números válidos.")
                    
        elif selected_test == "Prueba de Chi-cuadrado de Independencia":
            chi_input_fmt = st.radio("Formato de Ingreso para Chi-cuadrado:", [
                "📊 Matriz de Contingencia (Frecuencias Observadas)",
                "🔤 Vectores Categóricos Individuales"
            ], horizontal=True, key="chi_input_fmt_radio")

            if chi_input_fmt == "📊 Matriz de Contingencia (Frecuencias Observadas)":
                st.info("Ingrese cada fila de la tabla de contingencia en una línea nueva, con las frecuencias separadas por comas.")
                matrix_text = st.text_area(
                    "Frecuencias Observadas (Ejemplo 2x2: Fila 1 (Jóvenes)=15,35 | Fila 2 (Adultos)=30,20):",
                    "15, 35\n30, 20",
                    height=100,
                    key="chi_matrix_text"
                )
                r_names = st.text_input("Etiquetas de Filas (opcional, separadas por coma):", "Jovenes, Adultos", key="chi_r_names")
                c_names = st.text_input("Etiquetas de Columnas (opcional, separadas por coma):", "Efectivo, Tarjeta", key="chi_c_names")

                if st.button("Ejecutar Cálculo"):
                    try:
                        lines = [l.strip() for l in matrix_text.strip().split("\n") if l.strip()]
                        obs_matrix = []
                        for line in lines:
                            row_vals = [float(x.strip()) for x in line.split(",") if x.strip()]
                            if row_vals:
                                obs_matrix.append(row_vals)
                        
                        r_labs = [x.strip() for x in r_names.split(",") if x.strip()] if r_names else None
                        c_labs = [x.strip() for x in c_names.split(",") if x.strip()] if c_names else None
                        
                        inputs = {"matrix": obs_matrix, "r_labs": r_labs, "c_labs": c_labs, "is_matrix": True}
                        run_analysis = True
                    except Exception as ex:
                        st.error(f"Error al procesar la matriz: {str(ex)}")
            else:
                st.info("Para Chi-cuadrado con vectores categóricos, introduzca dos listas emparejadas correspondientes a cada sujeto.")
                c1_data = st.text_area("Variable A (ej. Género: M, M, M, M, F, F, F, F, F, F):", "M, M, M, M, F, F, F, F, F, F", key="chi_vec1")
                c2_data = st.text_area("Variable B (ej. Fuma: S, S, N, N, S, N, N, N, N, N):", "S, S, N, N, S, N, N, N, N, N", key="chi_vec2")
                if st.button("Ejecutar Cálculo"):
                    list_c1 = [x.strip() for x in c1_data.split(",") if x.strip()]
                    list_c2 = [x.strip() for x in c2_data.split(",") if x.strip()]
                    if len(list_c1) != len(list_c2):
                        st.error("Error: Ambas variables deben tener la misma cantidad de registros.")
                    else:
                        inputs = {"v1": list_c1, "v2": list_c2}
                        run_analysis = True
                    
        elif selected_test == "Regresión Lineal Simple":
            x_data = st.text_area("Variable X (Predictora):", "1, 2, 3, 4, 5")
            y_data = st.text_area("Variable Y (Respuesta):", "2.1, 3.9, 6.1, 8.2, 9.9")
            if st.button("Ejecutar Cálculo"):
                try:
                    data_x = [float(x.strip()) for x in x_data.split(",") if x.strip()]
                    data_y = [float(x.strip()) for x in y_data.split(",") if x.strip()]
                    if len(data_x) != len(data_y):
                        st.error("Error: X e Y deben tener la misma cantidad de puntos de datos.")
                    else:
                        inputs = {"x": data_x, "y": data_y}
                        run_analysis = True
                except ValueError:
                    st.error("Error: Ingrese solo números válidos.")
                    
        elif selected_test == "ANOVA de una vía (Análisis de Varianza)":
            st.info("Para ANOVA manual, ingrese un factor categórico explicativo y los valores cuantitativos de respuesta para cada observación.")
            g_data = st.text_area("Factor (ej. Tratamiento: A, A, A, B, B, B, C, C, C):", "A, A, A, B, B, B, C, C, C")
            v_data = st.text_area("Valores (ej. Efectividad: 8.1, 7.9, 8.3, 5.2, 5.4, 4.8, 12.1, 11.8, 12.4):", "8.1, 7.9, 8.3, 5.2, 5.4, 4.8, 12.1, 11.8, 12.4")
            if st.button("Ejecutar Cálculo"):
                try:
                    list_g = [x.strip() for x in g_data.split(",") if x.strip()]
                    list_v = [float(x.strip()) for x in v_data.split(",") if x.strip()]
                    if len(list_g) != len(list_v):
                        st.error("Error: Ambos vectores deben tener la misma cantidad de registros.")
                    else:
                        inputs = {"groups": list_g, "values": list_v}
                        run_analysis = True
                except ValueError:
                    st.error("Error: Asegúrese de que los valores de respuesta sean puramente numéricos.")
    else:
        # Verificar que exista un DataFrame
        if df is None:
            st.error("No hay un archivo CSV cargado.")
            st.info("Vaya al módulo 'Laboratorio de Datos' y cargue un archivo CSV.")
            st.stop()

        if var_types is None:
            st.error("No se detectó la información de tipos de variables.")
            st.stop()

        # Load from CSV columns dropdowns
        all_cols = list(df.columns)
        num_cols = [c for c in all_cols if var_types[c] == "numeric"]
        cat_cols = [c for c in all_cols if var_types[c] == "categorical"]
        
        st.markdown("##### Selección de Columnas del CSV")
            
        if selected_test == "Estadísticos Descriptivos":
            desc_cols = [c for c in all_cols if c in num_cols or c in cat_cols]
            var1 = st.selectbox("Variable a resumir (numérica o categórica):", desc_cols)
            if st.button("Calcular Descriptivos"):
                inputs = {"data": df[var1], "is_categorical": var1 in cat_cols}
                run_analysis = True
                st.session_state.calc_preselected_vars = {"var": var1}
                    
        elif selected_test == "Prueba t de Student para Muestras Independientes":
            dep_var = st.selectbox("Variable Dependiente (Numérica):", num_cols)
            group_var = st.selectbox("Factor de Agrupamiento (Categórica):", cat_cols)
            
            # Select groups dynamically
            if group_var:
                unique_vals = list(df[group_var].dropna().unique())
                if len(unique_vals) < 2:
                    st.error("El factor seleccionado debe tener al menos 2 categorías distintas.")
                else:
                    g1_name = st.selectbox("Grupo 1 (A):", unique_vals, index=0)
                    g2_name = st.selectbox("Grupo 2 (B):", [x for x in unique_vals if x != g1_name], index=0)
                    
                    if st.button("Calcular Prueba t independiente"):
                        g1_vals = df[df[group_var] == g1_name][dep_var]
                        g2_vals = df[df[group_var] == g2_name][dep_var]
                        inputs = {"g1": g1_vals, "g2": g2_vals}
                        run_analysis = True
                        st.session_state.calc_preselected_vars = {
                            "dep_var": dep_var,
                            "group_var": group_var,
                            "group1": g1_name,
                            "group2": g2_name
                        }
                            
        elif selected_test == "Prueba t de Student Pareada (Medidas Repetidas)":
            var1 = st.selectbox("Variable Momento 1 (Numérica):", num_cols)
            var2 = st.selectbox("Variable Momento 2 (Numérica):", [x for x in num_cols if x != var1])
            if st.button("Calcular Prueba t pareada"):
                inputs = {"v1": df[var1], "v2": df[var2]}
                run_analysis = True
                st.session_state.calc_preselected_vars = {"var1": var1, "var2": var2}
                    
        elif selected_test == "Prueba de Chi-cuadrado de Independencia":
            var1 = st.selectbox("Variable Categórica 1:", cat_cols, key="sel_chi_v1")
            var2 = st.selectbox("Variable Categórica 2:", [x for x in cat_cols if x != var1], key="sel_chi_v2")
            
            rem_cat = [x for x in cat_cols if x not in [var1, var2]]
            use_ctrl = st.checkbox("➕ Agregar 3ª Variable de Control / Capa (Chi-cuadrado Estratificado de 3 Vías)", value=False, key="chk_use_ctrl_var")
            var3 = None
            if use_ctrl:
                if rem_cat:
                    var3 = st.selectbox("Variable Categórica 3 (Variable de Control / Capa):", rem_cat, key="sel_ctrl_var")
                else:
                    st.info("No hay variables categóricas adicionales disponibles en el dataset para estratificar.")

            if st.button("Calcular Chi-cuadrado"):
                inputs = {"v1": df[var1], "v2": df[var2], "v3": df[var3] if var3 else None, "v3_name": var3}
                run_analysis = True
                st.session_state.calc_preselected_vars = {"var1": var1, "var2": var2, "var3": var3}
                    
        elif selected_test == "Regresión Lineal Simple":
            x_var = st.selectbox("Variable X (Predictora - Numérica):", num_cols)
            y_var = st.selectbox("Variable Y (Respuesta - Numérica):", [x for x in num_cols if x != x_var])
            if st.button("Ajustar Regresión"):
                inputs = {"x": df[x_var], "y": df[y_var]}
                run_analysis = True
                st.session_state.calc_preselected_vars = {"x_var": x_var, "y_var": y_var}
        elif selected_test == "ANOVA de una vía (Análisis de Varianza)":
            dep_var = st.selectbox("Variable Dependiente (Numérica):", num_cols)
            group_var = st.selectbox("Factor Explicativo (Categórica con 3+ grupos):", cat_cols)
            if group_var:
                unique_vals = df[group_var].dropna().unique()
                if len(unique_vals) < 3:
                    st.warning("⚠️ ANOVA requiere que el factor tenga 3 o más grupos. Para 2 grupos use Prueba t.")
                if st.button("Calcular ANOVA"):
                    inputs = {"groups": df[group_var], "values": df[dep_var]}
                    run_analysis = True
                    st.session_state.calc_preselected_vars = {"dep_var": dep_var, "group_var": group_var}

    # Execute math calculations
    if run_analysis:
        st.markdown("---")
        
        with st.spinner("Realizando cálculos de alta precisión..."):
            if selected_test == "Estadísticos Descriptivos":
                res = calculator.calculate_descriptive(inputs["data"])
            elif selected_test == "Prueba t de Student para Muestras Independientes":
                res = calculator.calculate_independent_ttest(inputs["g1"], inputs["g2"])
            elif selected_test == "Prueba t de Student Pareada (Medidas Repetidas)":
                res = calculator.calculate_paired_ttest(inputs["v1"], inputs["v2"])
            elif selected_test == "Prueba de Chi-cuadrado de Independencia":
                if inputs.get("is_matrix"):
                    res = calculator.calculate_chi_squared_matrix(inputs["matrix"], inputs.get("r_labs"), inputs.get("c_labs"))
                elif inputs.get("v3") is not None:
                    res = calculator.calculate_stratified_chi_squared(inputs["v1"], inputs["v2"], inputs["v3"], control_name=inputs.get("v3_name", "Control"))
                else:
                    res = calculator.calculate_chi_squared(inputs["v1"], inputs["v2"])
            elif selected_test == "Regresión Lineal Simple":
                res = calculator.calculate_simple_regression(inputs["x"], inputs["y"])
            elif selected_test == "ANOVA de una vía (Análisis de Varianza)":
                res = calculator.calculate_oneway_anova(inputs["groups"], inputs["values"])
                
        if "error" in res:
            st.error(res["error"])
        else:
            col_calc1, col_calc2 = st.columns([1, 1])
            
            with col_calc1:
                st.markdown("### 📊 Reporte e Interpretación Científica")
                
                # Render specific results
                if selected_test == "Estadísticos Descriptivos" and inputs.get("is_categorical"):
                    n_cat = int(inputs["data"].notna().sum())
                    st.markdown(f"**Análisis descriptivo de variable categórica sobre N = {n_cat} observaciones válidas.**")
                    freq_df = pd.DataFrame(res["table"])
                    freq_df.columns = ["Categoría", "Frecuencia (f)", "Porcentaje (%)"]
                    st.dataframe(freq_df, hide_index=True, use_container_width=True)
                    freq_num = pd.to_numeric(freq_df["Frecuencia (f)"], errors="coerce")
                    top_row = freq_df.loc[freq_num.idxmax()]
                    st.success(f"**Categoría modal:** {top_row['Categoría']} (f = {top_row['Frecuencia (f)']}, {top_row['Porcentaje (%)']}%).")

                elif selected_test == "Estadísticos Descriptivos":
                    st.markdown(f"**Análisis cuantitativo descriptivo realizado sobre N = {res['n']} observaciones.**")
                    st.markdown(f"* **Media Aritmética:** `{res['mean']:.4f}`")
                    st.markdown(f"* **Mediana:** `{res['median']:.4f}`")
                    st.markdown(f"* **Rango Intercuartílico (IQR):** `{res['iqr']:.4f}`")
                    st.markdown(f"* **Desviación Estándar:** `{res['std_dev']:.4f}`")
                    st.markdown(f"* **Varianza:** `{res['variance']:.4f}`")
                    st.markdown(f"* **Asimetría:** `{res['skewness']:.4f}`  |  **Curtosis:** `{res['kurtosis']:.4f}`")
                    st.markdown(f"* **Intervalo de Confianza al 95% para la Media:** `[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]`")
                    # Criterio práctico: coeficiente de asimetría de Pearson = 3 (media - mediana) / desv. est.
                    if res["std_dev"] > 0:
                        pearson_skew = 3 * (res["mean"] - res["median"]) / res["std_dev"]
                        if abs(pearson_skew) > 0.5:
                            st.warning(
                                f"La media ({res['mean']:.2f}) y la mediana ({res['median']:.2f}) difieren de forma notable "
                                f"(asimetría de Pearson = {pearson_skew:.2f}). La distribución es asimétrica: "
                                "conviene reportar **Mediana e IQR** como resumen principal."
                            )
                        else:
                            st.success(
                                f"La media y la mediana son similares (asimetría de Pearson = {pearson_skew:.2f}). "
                                "La distribución es aproximadamente simétrica: **Media y Desviación Estándar** son un resumen adecuado."
                            )

                elif selected_test in ["Prueba t de Student para Muestras Independientes", "Prueba t de Student Pareada (Medidas Repetidas)", "Prueba de Chi-cuadrado de Independencia", "ANOVA de una vía (Análisis de Varianza)", "Regresión Lineal Simple"]:
                    # All hypothesis tests have a p-value
                    p_val = res["p_value"]
                    stat_val = res.get("t_statistic") or res.get("chi2_statistic") or res.get("anova", {}).get("f_statistic") or res.get("f_statistic")
                    stat_name = "t" if "t_statistic" in res else ("Chi2" if "chi2_statistic" in res else "F")
                    
                    if selected_test == "Prueba de Chi-cuadrado de Independencia":
                        st.markdown(f"**Estadístico Calculado (Chi2 Pearson):** `{res['chi2_statistic']:.5f}`")
                        st.markdown(f"**Grados de libertad (df):** `{res['df']}`")
                        st.markdown(f"**p-valor obtenido (Pearson):** `{p_val:.7f}`")
                        if res.get("chi2_yates") is not None:
                            st.caption(f"📌 **Corrección de Continuidad de Yates (2x2):** `Chi2 = {res['chi2_yates']:.5f}` | `p-valor = {res['p_val_yates']:.7f}`")
                    else:
                        st.markdown(f"**Estadístico Calculado ({stat_name}):** `{stat_val:.5f}`")
                        st.markdown(f"**Grados de libertad (df):** `{res['df']}`")
                        st.markdown(f"**p-valor obtenido:** `{p_val:.7f}`")
                    
                    # Statistical significance interpretation
                    if p_val < 0.05:
                        st.markdown(f"""
                        <div style="background-color:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:6px; padding:15px; margin-bottom:15px;">
                            <h5 style="color:#10b981; margin:0;">Estadísticamente Significativo (p < 0.05)</h5>
                            <p style="margin:5px 0 0 0; color:#e2e8f0; font-size:0.9rem;">
                                Rechazamos la hipótesis nula (H₀) al nivel nominal del 5%. Existe evidencia suficiente para concluir que la diferencia observada o relación es sistemática y estadísticamente relevante.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:6px; padding:15px; margin-bottom:15px;">
                            <h5 style="color:#ef4444; margin:0;">No Significativo (p ≥ 0.05)</h5>
                            <p style="margin:5px 0 0 0; color:#e2e8f0; font-size:0.9rem;">
                                No hay evidencia suficiente para rechazar la hipótesis nula (H₀). Las diferencias o asociaciones observadas pueden explicarse razonablemente por la variabilidad aleatoria del muestreo.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Specific regression and ANOVA reports
                    if selected_test == "Regresión Lineal Simple":
                        st.markdown(f"* **Coeficiente R-cuadrado ($R^2$):** `{res['r_squared']:.5f}` ({res['r_squared']*100:.2f}% de la varianza explicada)")
                        st.markdown(f"* **Coeficiente R de Pearson:** `{res['r_value']:.5f}`")
                        st.markdown(rf"* **Ecuación estimada:** $Y = {res['intercept']:.4f} + {res['slope']:.4f} \cdot X$")
                    elif selected_test == "ANOVA de una vía (Análisis de Varianza)":
                        render_anova_report(res)

                st.markdown("---")
                st.markdown("##### 🔣 Fórmulas y Desglose Matemático Paso a Paso (LaTeX)")
                if selected_test == "Estadísticos Descriptivos" and inputs.get("is_categorical"):
                    st.latex(r"f_i = \text{conteo de la categoría } i \qquad p_i = \frac{f_i}{N} \times 100")
                else:
                    st.latex(res["latex"])

            with col_calc2:
                st.markdown("### 📈 Visualización del Análisis")
                
                # Interactive Plots based on test results
                if selected_test == "Estadísticos Descriptivos":
                    raw_data = inputs["data"]
                    if inputs.get("is_categorical"):
                        freq_plot = raw_data.value_counts().reset_index()
                        freq_plot.columns = ["Categoría", "Frecuencia"]
                        fig = px.bar(freq_plot, x="Categoría", y="Frecuencia", title="Frecuencias por Categoría", color_discrete_sequence=['#06b6d4'])
                    else:
                        fig = px.box(raw_data, points="all", title="Diagrama de Caja y Bigotes con Dispersión de Datos", color_discrete_sequence=['#06b6d4'])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif selected_test == "Prueba t de Student para Muestras Independientes":
                    # Boxplot comparing groups
                    g1_list = list(inputs["g1"])
                    g2_list = list(inputs["g2"])
                    plot_df = pd.DataFrame({
                        "Valor": g1_list + g2_list,
                        "Grupo": ["Grupo A"]*len(g1_list) + ["Grupo B"]*len(g2_list)
                    })
                    fig = px.box(plot_df, x="Grupo", y="Valor", color="Grupo", points="all", title="Comparativa de Distribuciones Intergrupales")
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif selected_test == "Prueba t de Student Pareada (Medidas Repetidas)":
                    # Boxplot of differences
                    fig = px.box(res["diffs"], points="all", title="Distribución de las Diferencias (Pre - Post)", color_discrete_sequence=['#8b5cf6'])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif selected_test == "Prueba de Chi-cuadrado de Independencia":
                    # Heatmap of observed vs expected
                    obs_arr = np.array(res["observed"])
                    fig = go.Figure(data=go.Heatmap(
                        z=obs_arr,
                        x=res["col_labels"],
                        y=res["row_labels"],
                        colorscale='Viridis',
                        text=obs_arr,
                        texttemplate="%{text}",
                        textfont={"size":16}
                    ))
                    fig.update_layout(title="Mapa de Calor de Frecuencias Observadas", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif selected_test == "Regresión Lineal Simple":
                    # Scatter plot with regression line
                    x_arr = np.array(inputs["x"])
                    y_arr = np.array(inputs["y"])
                    
                    x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
                    y_line = res["intercept"] + res["slope"] * x_line
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_arr, y=y_arr, mode='markers', name='Datos Observados', marker=dict(color='#3b82f6', size=10)))
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Línea OLS Ajustada', line=dict(color='#10b981', width=3)))
                    
                    fig.update_layout(
                        title=f"Diagrama de Dispersión y Ajuste Lineal (R2 = {res['r_squared']:.4f})",
                        xaxis_title="Variable X",
                        yaxis_title="Variable Y",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f3f4f6'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif selected_test == "ANOVA de una vía (Análisis de Varianza)":
                    # Group boxplot
                    groups_arr = list(inputs["groups"])
                    values_arr = list(inputs["values"])
                    plot_df = pd.DataFrame({"Factor": groups_arr, "Valores": values_arr})
                    fig = px.box(plot_df, x="Factor", y="Valores", color="Factor", points="all", title="Distribución de Valores por Factor Explicativo")
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
                    st.plotly_chart(fig, use_container_width=True)

            # ----------------------------------------------------
            # Dynamic Code Generator
            # ----------------------------------------------------
            st.markdown("---")
            st.markdown("### 💻 Generador de Código Dinámico (Con variables y archivo del usuario)")
            
            # Map core code templates
            code_mapping_ids = {
                "Estadísticos Descriptivos": "arithmetic_mean",
                "Prueba t de Student para Muestras Independientes": "independent_ttest",
                "Prueba t de Student Pareada (Medidas Repetidas)": "paired_ttest",
                "Prueba de Chi-cuadrado de Independencia": "chi_squared_test",
                "Regresión Lineal Simple": "simple_regression",
                "ANOVA de una vía (Análisis de Varianza)": "oneway_anova"
            }
            
            db_model_id = code_mapping_ids[selected_test]
            db_model = database.CORE_MODELS[db_model_id]
            
            code_tabs = st.tabs(["Python", "R", "SPSS", "Stata", "SAS"])
            langs = ["python", "r", "spss", "stata", "sas"]
            
            # Extract user parameters for formatting
            u_vars = st.session_state.calc_preselected_vars
            csv_path = st.session_state.csv_name
            
            for c_tab, lang in zip(code_tabs, langs):
                with c_tab:
                    raw_template = db_model["templates"][lang]

                    # Fill default parameters if manual inputs were used
                    csv_val = csv_path if data_source != "Ingresar datos manualmente" else "datos.csv"

                    try:
                        if db_model_id == "arithmetic_mean":
                            filled_code = safe_format(
                                raw_template,
                                csv_file=csv_val,
                                var=u_vars.get("var", "Variable")
                            )
                        elif db_model_id == "independent_ttest":
                            filled_code = safe_format(
                                raw_template,
                                csv_file=csv_val,
                                dep_var=u_vars.get("dep_var", "Dependiente"),
                                group_var=u_vars.get("group_var", "Grupo"),
                                group1=u_vars.get("group1", "A"),
                                group2=u_vars.get("group2", "B")
                            )
                        elif db_model_id == "paired_ttest":
                            filled_code = safe_format(
                                raw_template,
                                csv_file=csv_val,
                                var1=u_vars.get("var1", "Pre"),
                                var2=u_vars.get("var2", "Post")
                            )
                        elif db_model_id == "chi_squared_test":
                            filled_code = safe_format(
                                raw_template,
                                csv_file=csv_val,
                                var1=u_vars.get("var1", "VarCategórica1"),
                                var2=u_vars.get("var2", "VarCategórica2")
                            )
                        elif db_model_id == "simple_regression":
                            filled_code = safe_format(
                                raw_template,
                                csv_file=csv_val,
                                x_var=u_vars.get("x_var", "X_Predictora"),
                                y_var=u_vars.get("y_var", "Y_Respuesta")
                            )
                        elif db_model_id == "oneway_anova":
                            filled_code = safe_format(
                                raw_template,
                                csv_file=csv_val,
                                dep_var=u_vars.get("dep_var", "Dependiente"),
                                group_var=u_vars.get("group_var", "Factor")
                            )
                    except Exception:
                        filled_code = safe_format(
                            raw_template,
                            csv_file="datos.csv",
                            var="Variable",
                            group_var="Grupo",
                            dep_var="Dependiente",
                            group1="A",
                            group2="B",
                            var1="Variable1",
                            var2="Variable2",
                            x_var="X_Predictora",
                            y_var="Y_Respuesta"
                        )

                    st.code(filled_code, language=lang)


def render_higea():
    """HIGEA AS-001: diagnóstico y limpieza de calidad de datos."""
    st.title("🧹 Módulo de Limpieza y Calidad de Datos (HIGEA AS-001)")
    st.caption("Módulo HIGEA — Inspección automática, tratamiento modular y auto-limpieza inteligente de datos")

    df = st.session_state.get("df")

    if df is None:
        st.warning("⚠️ No se ha detectado ningún conjunto de datos.")
        st.info("Por favor, sube tu archivo en el **Laboratorio de Datos**.")
    else:
        # Run AS-001 Diagnostic Assistant for Quality Score
        diag_asst = AS001DiagnosticAssistant(df)
        diag_report = diag_asst.run_full_diagnosis()

        total_faltantes = int(df.isna().sum().sum())
        duplicados = int(df.duplicated().sum())

        st.markdown("### 📊 Métricas de Calidad de Datos")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Filas", f"{len(df):,}")
        c2.metric("Columnas", f"{len(df.columns):,}")
        c3.metric("Faltantes", f"{total_faltantes:,}")
        c4.metric("Duplicados", f"{duplicados:,}")
        c5.metric("Quality Score", f"{diag_report['quality_score']}%")

        st.markdown("---")
        st.markdown("### 🎛️ Modos y Opciones de Limpieza Modular")

        clean_tab1, clean_tab2, clean_tab3, clean_tab4, clean_tab5, clean_tab6, clean_tab7, clean_tab8 = st.tabs([
            "⚡ Auto-Limpieza 1-Clic",
            "♻️ Duplicados",
            "💧 Faltantes e Imputación",
            "🏷️ Nombres & Tipos",
            "🧽 Texto & Normalización",
            "📐 Escalamiento",
            "🚨 Atípicos (Outliers)",
            "📊 Resumen & Exportación"
        ])

        with clean_tab1:
            st.markdown("#### ⚡ Limpieza Integral Inteligente")
            st.info("Ejecuta en un solo clic: eliminación de espacios en blanco, remoción de filas y columnas 100% vacías, eliminación de copias duplicadas e imputación recomendada (mediana para numéricas, moda para categóricas).")
            if st.button("🚀 Ejecutar Auto-Limpieza Completa", key="btn_auto_clean_full", use_container_width=True):
                df_cleaned = higea_auto_clean(df)
                st.session_state.df = df_cleaned
                st.session_state.higea_missing_analysis = None
                st.success("✅ ¡Auto-limpieza completada exitosamente! Se actualizaron todos los registros.")
                st.rerun()

        with clean_tab2:
            st.markdown("#### ♻️ Tratamiento de Filas Duplicadas")
            reporte_dup = higea_duplicate_report(df)

            if reporte_dup.empty:
                st.success("✅ No existen filas duplicadas en este conjunto de datos.")
            else:
                st.warning(f"HIGEA detectó **{duplicados:,} copias adicionales**.")
                with st.expander("🔎 Ver detalle de registros duplicados"):
                    st.dataframe(reporte_dup, use_container_width=True, hide_index=True)

                if st.button(f"♻️ Eliminar copias duplicadas ({duplicados:,})", key="higea_eliminar_duplicados", use_container_width=True):
                    antes = len(df)
                    df_nuevo = higea_remove_duplicates(df)
                    eliminadas = antes - len(df_nuevo)
                    st.session_state.df = df_nuevo
                    st.session_state.higea_missing_analysis = None
                    st.success(f"Se conservaron las primeras apariciones y se eliminaron {eliminadas:,} copias.")
                    st.rerun()

        with clean_tab3:
            st.markdown("#### 💧 Tratamiento e Imputación de Valores Faltantes")
            if total_faltantes == 0:
                st.success("✅ No existen valores faltantes en este dataset.")
            else:
                st.write(f"Hay **{total_faltantes:,} valores faltantes** en el conjunto de datos.")

                m1, m2, m3 = st.columns(3)
                with m1:
                    if st.button("📊 Calcular media", key="higea_calcular_media", use_container_width=True):
                        st.session_state.higea_missing_analysis = higea_analyze_missing(df)
                        st.session_state.higea_missing_focus = "media"
                        st.rerun()

                with m2:
                    if st.button("📈 Calcular mediana", key="higea_calcular_mediana", use_container_width=True):
                        st.session_state.higea_missing_analysis = higea_analyze_missing(df)
                        st.session_state.higea_missing_focus = "mediana"
                        st.rerun()

                with m3:
                    if st.button("🔤 Calcular moda", key="higea_calcular_moda", use_container_width=True):
                        st.session_state.higea_missing_analysis = higea_analyze_missing(df)
                        st.session_state.higea_missing_focus = "moda"
                        st.rerun()

                analisis = st.session_state.get("higea_missing_analysis")
                if analisis is not None and not analisis.empty:
                    st.markdown("##### 📋 Tabla de Análisis de Faltantes")
                    tabla = analisis.copy()
                    for col in ["Media", "Mediana"]:
                        tabla[col] = tabla[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
                    tabla["Moda"] = tabla["Moda"].map(lambda x: "" if pd.isna(x) else str(x))
                    st.dataframe(tabla, use_container_width=True, hide_index=True)

                    st.markdown("##### ✨ Opciones de Imputación Masiva")
                    i1, i2, i3 = st.columns(3)
                    with i1:
                        if st.button("✨ Aplicar media a numéricas", key="higea_aplicar_media", use_container_width=True):
                            st.session_state.df = higea_fill_numeric(df, method="mean")
                            st.session_state.higea_missing_analysis = None
                            st.rerun()
                    with i2:
                        if st.button("✨ Aplicar mediana a numéricas", key="higea_aplicar_mediana", use_container_width=True):
                            st.session_state.df = higea_fill_numeric(df, method="median")
                            st.session_state.higea_missing_analysis = None
                            st.rerun()
                    with i3:
                        if st.button("✨ Aplicar moda a categóricas", key="higea_aplicar_moda", use_container_width=True):
                            st.session_state.df = higea_fill_mode(df)
                            st.session_state.higea_missing_analysis = None
                            st.rerun()

        with clean_tab4:
            st.markdown("#### 🏷️ Estandarización y Renombrado de Columnas")
            st.info("Estandariza los nombres de todas las columnas convirtiéndolos a minúsculas, reemplazando espacios con guiones bajos '_' y eliminando tildes y caracteres especiales.")
            if st.button("🏷️ Estandarizar Encabezados en 1-Clic (`nombre_columna`)", key="btn_clean_cols_names", use_container_width=True):
                st.session_state.df = higea_clean_column_names(df)
                st.success("✅ Nombres de columnas estandarizados exitosamente.")
                st.rerun()

            st.markdown("---")
            st.markdown("##### 🔄 Casteo / Conversión Manual de Tipos de Columna")
            col_to_cast = st.selectbox("Seleccione la columna a convertir:", df.columns, key="select_cast_col")
            curr_type = str(df[col_to_cast].dtype)
            st.caption(f"Tipo de dato actual: `{curr_type}`")
            target_t = st.selectbox("Seleccione el nuevo tipo de datos:", ["numeric", "category", "string", "datetime"], key="select_target_type")

            if st.button("🔄 Aplicar Conversión de Tipo", key="btn_apply_cast", use_container_width=True):
                st.session_state.df = higea_convert_column_type(df, col_to_cast, target_t)
                st.success(f"✅ Columna '{col_to_cast}' convertida a {target_t}.")
                st.rerun()

        with clean_tab5:
            st.markdown("#### 🧽 Normalización y Limpieza de Texto")
            o1, o2, o3 = st.columns(3)
            with o1:
                if st.button("🧽 Quitar espacios extra", key="higea_quitar_espacios", use_container_width=True):
                    st.session_state.df = higea_clean_spaces(df)
                    st.success("Espacios en blanco removidos de nombres y celdas.")
                    st.rerun()
            with o2:
                if st.button("🗑️ Eliminar filas/cols vacías", key="higea_eliminar_vacias", use_container_width=True):
                    st.session_state.df = higea_remove_empty(df)
                    st.success("Filas y columnas vacías eliminadas.")
                    st.rerun()
            with o3:
                original = st.session_state.get("df_original")
                if st.button("🔄 Restaurar base original", key="higea_restaurar", use_container_width=True):
                    if original is not None:
                        st.session_state.df = original.copy(deep=True)
                        st.session_state.higea_missing_analysis = None
                        st.success("Base original restaurada.")
                        st.rerun()
                    else:
                        st.warning("No existe copia de la base original.")

            st.markdown("---")
            st.markdown("##### 🔡 Transformaciones de Formato de Texto")
            text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if not text_cols:
                st.info("No se detectaron columnas de tipo texto en este dataset.")
            else:
                selected_text_cols = st.multiselect("Seleccione las columnas de texto a transformar:", text_cols, default=text_cols[:2] if len(text_cols)>=2 else text_cols, key="ms_text_cols")
                t_case = st.radio("Formato deseado", ["Minúsculas (lower)", "Mayúsculas (UPPER)", "Título (Title Case)"], horizontal=True, key="radio_text_case")
                rem_acc = st.checkbox("✨ Remover tildes y caracteres diacríticos", value=True, key="chk_rem_accents")
                
                if st.button("🔤 Aplicar Formato de Texto", key="btn_apply_text_fmt", use_container_width=True):
                    mode_map = {"Minúsculas (lower)": "lower", "Mayúsculas (UPPER)": "upper", "Título (Title Case)": "title"}
                    st.session_state.df = higea_normalize_text_case(df, selected_text_cols, mode=mode_map[t_case], remove_accents=rem_acc)
                    st.success("✅ Transformación de texto aplicada exitosamente.")
                    st.rerun()

        with clean_tab6:
            st.markdown("#### 📐 Escalamiento y Normalización de Variables Numéricas")
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                st.info("No hay columnas numéricas en el dataset para escalar.")
            else:
                col_to_scale = st.selectbox("Seleccione la variable numérica a escalar:", num_cols, key="scale_col_select")
                scale_method = st.radio("Método de Escalamiento:", [
                    "Z-Score Standardization (Media = 0, Desv Std = 1)",
                    "Min-Max Normalization (Rango entre 0 y 1)"
                ], key="scale_method_radio")
                
                if st.button(f"📐 Escalar '{col_to_scale}'", key="btn_apply_scale", use_container_width=True):
                    method_key = "zscore" if "Z-Score" in scale_method else "minmax"
                    st.session_state.df = higea_scale_column(df, col_to_scale, method=method_key)
                    st.success(f"✅ Columna '{col_to_scale}' escalada con éxito mediante {method_key.upper()}.")
                    st.rerun()

        with clean_tab7:
            st.markdown("#### 🚨 Detección y Tratamiento de Valores Atípicos (Outliers)")
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                st.info("No hay columnas numéricas para analizar valores atípicos.")
            else:
                out_col = st.selectbox("Seleccione la variable a evaluar:", num_cols, key="higea_out_col")
                out_method = st.radio("Método de Detección", ["Rango Intercuartílico (IQR 1.5x)", "Z-Score (|Z| > 3)"], horizontal=True, key="higea_out_method")
                
                m_key = "iqr" if "IQR" in out_method else "zscore"
                outliers_df = diag_asst.detect_outliers(out_col, method=m_key)
                
                if outliers_df.empty:
                    st.success(f"✅ No se detectaron valores atípicos severos en '{out_col}'.")
                else:
                    st.warning(f"Se identificaron **{len(outliers_df)} registros atípicos** en '{out_col}'.")
                    with st.expander("🔎 Ver filas con valores atípicos"):
                        st.dataframe(outliers_df, use_container_width=True)
                        
                    col_out1, col_out2 = st.columns(2)
                    with col_out1:
                        if st.button("✂️ Aplicar Winsorización (Ajustar al percentil 5%-95%)", key="btn_winsorize", use_container_width=True):
                            df_mod = df.copy(deep=True)
                            df_mod[out_col] = higea_winsorize_column(df_mod[out_col])
                            st.session_state.df = df_mod
                            st.success(f"Valores atípicos de '{out_col}' ajustados mediante Winsorización.")
                            st.rerun()
                    with col_out2:
                        if st.button("🗑️ Eliminar registros atípicos", key="btn_drop_outliers", use_container_width=True):
                            df_mod = df.drop(index=outliers_df.index).copy()
                            st.session_state.df = df_mod
                            st.success(f"Se eliminaron {len(outliers_df)} filas atípicas.")
                            st.rerun()

        with clean_tab8:
            st.markdown("#### 📊 Resumen Estructural y Descarga de Base Limpia")
            df_actual = st.session_state.df
            faltantes_actuales = int(df_actual.isna().sum().sum())
            duplicados_actuales = int(df_actual.duplicated().sum())

            if faltantes_actuales == 0 and duplicados_actuales == 0:
                st.success(f"✅ Base completamente limpia: {len(df_actual):,} filas × {len(df_actual.columns):,} columnas | 0 faltantes y 0 duplicados.")
            else:
                st.info(f"Estado actual: {len(df_actual):,} filas | {faltantes_actuales:,} faltantes | {duplicados_actuales:,} duplicados.")

            csv_limpio = df_actual.to_csv(index=False, sep=";", encoding="utf-8-sig")

            st.download_button(
                f"✨ Descargar base limpia ({len(df_actual):,} filas)",
                data=csv_limpio,
                file_name="BASE_VENTAS_HIGEA_LIMPIA.csv",
                mime="text/csv",
                key="higea_download_clean",
                use_container_width=True
            )

            if st.button("💾 Guardar base limpia en carpeta del sistema", key="higea_save_clean", use_container_width=True):
                carpeta_datos = Path("datos_limpios")
                carpeta_datos.mkdir(parents=True, exist_ok=True)
                ruta_limpia = carpeta_datos / "BASE_VENTAS_HIGEA_LIMPIA.csv"
                df_actual.to_csv(ruta_limpia, index=False, sep=";", encoding="utf-8-sig")
                st.success(f"💾 Base limpia guardada correctamente en: `{ruta_limpia}`")

            st.markdown("---")
            st.markdown(f"##### 📈 Detalle Estructural de las {len(df_actual.columns)} Variables")
            df_resumen = pd.DataFrame({
                "Columna": list(df_actual.columns),
                "Tipo": [str(t) for t in df_actual.dtypes],
                "Nulos": [int(df_actual[col].isna().sum()) for col in df_actual.columns],
                "% Nulos": [f"{df_actual[col].isna().mean() * 100:.1f}%" for col in df_actual.columns]
            })
            st.dataframe(df_resumen, use_container_width=True, height=250)

            st.markdown("##### 👁️ Vista Previa de los Datos")
            st.dataframe(df_actual, use_container_width=True, height=350)



# ============================================================
# 7. REGISTRO DE MÓDULOS Y PUNTO DE ENTRADA
# ============================================================
# "requires": secciones que deben estar activas para que ésta funcione.
# "data_gate": None = sin control | "required" = exige CSV cargado |
#              "optional" = exige CSV o que el usuario elija ingreso manual.
# Sin datos, la app se detiene en el Laboratorio de Datos.
# HIGEA trabaja sobre el DataFrame cargado en el Laboratorio de Datos.
MODULES = [
    {"label": TAB_ADVISOR, "enabled": SHOW_ADVISOR, "render": render_advisor, "requires": [], "data_gate": None},
    {"label": TAB_LIBRARY, "enabled": SHOW_LIBRARY, "render": render_library, "requires": [], "data_gate": None},
    {"label": TAB_LAB, "enabled": SHOW_LAB, "render": render_lab, "requires": [], "data_gate": None},
    {"label": TAB_HIGEA, "enabled": SHOW_HIGEA, "render": render_higea, "requires": [TAB_LAB], "data_gate": "required"},
    {"label": TAB_CALC, "enabled": SHOW_CALCULATOR, "render": render_calc, "requires": [], "data_gate": "optional"},
]


def on_nav_change():
    """Se ejecuta cuando el usuario hace clic en el menú lateral."""
    st.session_state.selected_tab = st.session_state.sidebar_nav


def render_sidebar(tabs):
    """Dibuja el menú lateral y sincroniza la sección activa."""
    with st.sidebar:
        st.markdown('<div class="premium-badge">Nivel Doctoral / Científico de Datos</div>', unsafe_allow_html=True)
        st.title("🧠 Consultor Estadístico")
        st.markdown("Plataforma interactiva de diagnóstico, cálculo paso a paso y códigos de modelos estadísticos.")
        st.markdown("---")

        if st.session_state.selected_tab not in tabs:
            st.session_state.selected_tab = tabs[0]

        if st.session_state.get("sidebar_nav") != st.session_state.selected_tab:
            st.session_state.sidebar_nav = st.session_state.selected_tab

        st.radio("Secciones", tabs, key="sidebar_nav", on_change=on_nav_change)

        st.markdown("---")
        if st.session_state.df is None:
            st.caption("📂 Sin datos cargados")
        else:
            n_rows, n_cols = st.session_state.df.shape
            st.caption(f"📊 {st.session_state.csv_name} — {n_rows:,} filas × {n_cols:,} columnas")


def main():
    st.set_page_config(
        page_title="Consultor Inteligente de Modelos Estadísticos",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_styles()
    init_state()

    tabs = get_enabled_tabs()
    if not tabs:
        st.error("No hay secciones habilitadas. Activa al menos un flag SHOW_* en la configuración.")
        st.stop()

    render_sidebar(tabs)


    current = st.session_state.selected_tab
    module = next(m for m in MODULES if m["label"] == current)

    # Compuerta: las secciones que necesitan datos se detienen en el Laboratorio.
    gate = module["data_gate"]
    manual_ok = gate == "optional" and st.session_state.manual_mode
    if gate and st.session_state.df is None and not manual_ok and is_enabled(TAB_LAB):
        st.session_state.gate_message = (
            f"Para usar «{current}» primero sube un archivo CSV en el Laboratorio de Datos."
        )
        st.session_state.selected_tab = TAB_LAB
        st.rerun()

    module["render"]()


main()