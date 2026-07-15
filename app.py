# -*- coding: utf-8 -*-
"""
Main Streamlit application for 'Consultor Inteligente de Modelos Estadísticos'.
"""

# Hola




import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import database
import advisor
import calculator
import data_processor
import tests
import string

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

# Configure Streamlit page
st.set_page_config(
    page_title="Consultor Inteligente de Modelos Estadísticos",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Run math validations once at startup
if "test_results" not in st.session_state:
    st.session_state.test_results = tests.run_mathematical_tests()

# State initialization
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "wizard_answers" not in st.session_state:
    st.session_state.wizard_answers = {}
if "df" not in st.session_state:
    st.session_state.df = None
if "var_types" not in st.session_state:
    st.session_state.var_types = {}
if "csv_name" not in st.session_state:
    st.session_state.csv_name = "datos.csv"
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Asesor Metodológico"
if "calc_test" not in st.session_state:
    st.session_state.calc_test = "Estadísticos Descriptivos"
if "calc_preselected_vars" not in st.session_state:
    st.session_state.calc_preselected_vars = {}

inject_custom_styles()

# Sidebar Layout
with st.sidebar:
    st.markdown('<div class="premium-badge">Nivel Doctoral / Científico de Datos</div>', unsafe_allow_html=True)
    st.title("🧠 Consultor Estadístico")
    st.markdown("Plataforma interactiva de diagnóstico, cálculo paso a paso y códigos de modelos estadísticos.")
    st.markdown("---")
    
    # Navigation
    tabs = ["Asesor Metodológico", "Biblioteca de Modelos", "Laboratorio de Datos", "Calculadora Analítica"]
    
    # Simple trick to handle navigation programmatically
    nav_index = tabs.index(st.session_state.selected_tab)
    selected_tab = st.radio("Secciones", tabs, index=nav_index, key="sidebar_nav")
    if selected_tab != st.session_state.selected_tab:
        st.session_state.selected_tab = selected_tab
        st.rerun()
        
    st.markdown("---")
    
    # Mathematical check indicator
    if st.session_state.test_results["all_pass"]:
        st.markdown(
            '<div class="math-check-badge"><span class="dot"></span>Rigor Matemático Validado (100% PASS)</div>', 
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Rigor Matemático: Algunas pruebas fallaron")
        
    st.markdown("<br><small style='color:#64748b;'>Desarrollado para la privacidad de sus datos. Todos los cálculos se realizan localmente en memoria.</small>", unsafe_allow_html=True)

# Main Dashboard Routing
st.session_state.selected_tab = selected_tab

if st.session_state.selected_tab == "Asesor Metodológico":
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
        
        selected_opt = st.radio("Seleccione el objetivo de su investigación:", opts_labels, index=curr_idx)
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    elif step == 2:
        q_id = "var_type"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el tipo de variables:", opts_labels, index=curr_idx)
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    elif step == 3:
        q_id = "var_count"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el número de variables implicadas:", opts_labels, index=curr_idx)
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    elif step == 4:
        q_id = "independence"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el tipo de relación entre muestras:", opts_labels, index=curr_idx)
        answers[q_id] = opts_keys[opts_labels.index(selected_opt)]
        
    elif step == 5:
        q_id = "assumptions"
        q_data = advisor.QUESTIONS[q_id]
        st.markdown(f'<div class="glass-panel border-blue"><h4>{q_data["title"]}</h4><p>{q_data["desc"]}</p></div>', unsafe_allow_html=True)
        
        opts_labels = [opt[1] for opt in q_data["options"]]
        opts_keys = [opt[0] for opt in q_data["options"]]
        curr_val = answers.get(q_id, opts_keys[0])
        curr_idx = opts_keys.index(curr_val) if curr_val in opts_keys else 0
        
        selected_opt = st.radio("Seleccione el cumplimiento de supuestos de normalidad/varianza:", opts_labels, index=curr_idx)
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
            
            # Action to send to calculator if core
            core_mappings = {
                "arithmetic_mean": "Estadísticos Descriptivos",
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
                if st.button(f"🚀 Cargar '{model['name']}' en el Módulo Calculadora"):
                    st.session_state.calc_test = cal_test_name
                    st.session_state.selected_tab = "Calculadora Analítica"
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
                
                if st.button("Ver Alternativa"):
                    # Temporarily replace with alternative recommendation
                    # A trick to swap
                    answers["assumptions"] = "nonparametric" if answers.get("assumptions") == "parametric" else "parametric"
                    st.rerun()
                    
            if st.button("🔄 Reiniciar Asesor"):
                st.session_state.wizard_step = 1
                st.session_state.wizard_answers = {}
                st.rerun()

elif st.session_state.selected_tab == "Biblioteca de Modelos":
    st.title("📚 Biblioteca Científica de Modelos")
    st.subheader("Catálogo de nivel doctoral con más de 300 modelos y lenguajes de programación estadística.")
    
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
        query = search_query.lower()
        catalog = [m for m in catalog if query in m["name"].lower() or query in m["desc"].lower()]
        
    st.markdown(f"**Se encontraron {len(catalog)} modelos estadísticos en este segmento.**")
    
    # Page pagination / grid
    # To avoid rendering 300+ items at once, limit to 20 per page
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
            with st.expander(f"📖 Ver Fórmulas, Supuestos y Plantillas de Código en 5 Lenguajes"):
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
                
                # Render code templates if available
                if m["templates"]:
                    st.markdown("---")
                    st.markdown("#### Plantillas de Código Multilingüe (Variables Genéricas)")
                    code_tabs = st.tabs(["Python", "R", "SPSS", "Stata", "SAS"])
                    langs = ["python", "r", "spss", "stata", "sas"]
                    
                    for t_tab, lang in zip(code_tabs, langs):
                        with t_tab:
                            # Render code block
                            template_code = m["templates"][lang]
                            # Clean print placeholders for formatting visual blocks
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

elif st.session_state.selected_tab == "Laboratorio de Datos":
    st.title("🧪 Laboratorio de Análisis de Datos")
    st.subheader("Importe archivos CSV y analice descriptivamente variables de forma instantánea.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Subir Archivo de Datos (CSV)", type=["csv"], help="Formatos con separador por coma (,) o punto y coma (;).")
    
    if uploaded_file is not None:
        try:
            # Parse CSV
            df = data_processor.parse_csv(uploaded_file)
            # Detect variable types
            cleaned_df, var_types = data_processor.detect_variable_types(df)
            
            # Store in session state
            st.session_state.df = cleaned_df
            st.session_state.var_types = var_types
            st.session_state.csv_name = uploaded_file.name
            
            st.success(f"✅ Archivo '{uploaded_file.name}' cargado exitosamente. Se detectaron {df.shape[0]} filas y {df.shape[1]} columnas.")
        except Exception as e:
            st.error(f"Error al procesar el archivo CSV: {str(e)}")
            
    # Display statistics if data is loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        var_types = st.session_state.var_types
        
        st.markdown("### Tabla Vista Previa (Primeras 10 filas)")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Análisis Descriptivo y Visual de Variables")
        
        # Select variable
        all_cols = list(df.columns)
        selected_col = st.selectbox("Seleccione la variable a analizar:", all_cols)
        
        if selected_col:
            col_type = var_types[selected_col]
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
                if col_type == "numeric":
                    # Histograma Plotly
                    fig = px.histogram(
                        df, x=selected_col, 
                        marginal="box", 
                        title=f"Distribución e Histograma de {selected_col}",
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f3f4f6'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Bar plot Plotly
                    freq_df = pd.DataFrame(stats_res["table"])
                    fig = px.bar(
                        freq_df, x="value", y="count",
                        title=f"Frecuencias de Categorías para {selected_col}",
                        color="value",
                        labels={"value": "Categoría", "count": "Frecuencia"},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f3f4f6'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Por favor, suba un archivo CSV en el panel de arriba para iniciar el Laboratorio de Datos.")

elif st.session_state.selected_tab == "Calculadora Analítica":
    st.title("🧮 Calculadora de Modelos Estadísticos")
    st.subheader("Cálculo matemático paso a paso con fórmulas LaTeX e interpretación científica.")
    
    # Selected test
    tests_list = [
        "Estadísticos Descriptivos",
        "Prueba t de Student para Muestras Independientes",
        "Prueba t de Student Pareada (Medidas Repetidas)",
        "Prueba de Chi-cuadrado de Independencia",
        "Regresión Lineal Simple",
        "ANOVA de una vía (Análisis de Varianza)"
    ]
    
    selected_test = st.selectbox("Seleccione el análisis estadístico a realizar:", tests_list, index=tests_list.index(st.session_state.calc_test))
    st.session_state.calc_test = selected_test
    
    # Source switcher (Manual vs CSV)
    data_source = "Manual"
    if st.session_state.df is not None:
        data_source = st.radio("Fuente de Datos", ["Seleccionar del CSV cargado", "Ingresar datos manualmente"], horizontal=True)
        
    df = st.session_state.df
    var_types = st.session_state.var_types
    
    # Render Inputs conditionally
    run_analysis = False
    inputs = {}
    
    if data_source == "Ingresar datos manualmente":
        st.markdown("##### Ingreso de Datos Manuales (valores numéricos separados por comas)")
        
        if selected_test == "Estadísticos Descriptivos":
            manual_data = st.text_area("Datos:", "10, 15, 12, 18, 20, 14, 16, 22, 19, 11")
            if st.button("Ejecutar Cálculo"):
                try:
                    data_list = [float(x.strip()) for x in manual_data.split(",") if x.strip()]
                    inputs = {"data": data_list}
                    run_analysis = True
                except ValueError:
                    st.error("Error: Asegúrese de ingresar solo números separados por comas.")
                    
        elif selected_test == "Prueba t de Student para Muestras Independientes":
            g1_data = st.text_area("Grupo A:", "12.5, 11.2, 14.1, 13.0, 12.2")
            g2_data = st.text_area("Grupo B:", "15.1, 16.2, 14.5, 15.8, 14.9")
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
            st.info("Para Chi-cuadrado manual, introduciremos dos vectores categóricos emparejados correspondientes a cada sujeto.")
            c1_data = st.text_area("Variable A (ej. Género: M, M, M, M, F, F, F, F, F, F):", "M, M, M, M, F, F, F, F, F, F")
            c2_data = st.text_area("Variable B (ej. Fuma: S, S, N, N, S, N, N, N, N, N):", "S, S, N, N, S, N, N, N, N, N")
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
        # Load from CSV columns dropdowns
        all_cols = list(df.columns)
        num_cols = [c for c in all_cols if var_types[c] == "numeric"]
        cat_cols = [c for c in all_cols if var_types[c] == "categorical"]
        
        st.markdown("##### Selección de Columnas del CSV")
        
        if selected_test == "Estadísticos Descriptivos":
            var1 = st.selectbox("Variable numérica a resumir:", num_cols)
            if st.button("Calcular Descriptivos"):
                inputs = {"data": df[var1]}
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
            var1 = st.selectbox("Variable Categórica 1:", cat_cols)
            var2 = st.selectbox("Variable Categórica 2:", [x for x in cat_cols if x != var1])
            if st.button("Calcular Chi-cuadrado"):
                inputs = {"v1": df[var1], "v2": df[var2]}
                run_analysis = True
                st.session_state.calc_preselected_vars = {"var1": var1, "var2": var2}
                
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
                if selected_test == "Estadísticos Descriptivos":
                    st.markdown(f"**Análisis cuantitativo descriptivo realizado sobre N = {res['n']} observaciones.**")
                    st.markdown(f"* **Media Aritmética:** `{res['mean']:.4f}`")
                    st.markdown(f"* **Desviación Estándar:** `{res['std_dev']:.4f}`")
                    st.markdown(f"* **Varianza:** `{res['variance']:.4f}`")
                    st.markdown(f"* **Intervalo de Confianza al 95% para la Media:** `[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]`")
                    
                elif selected_test in ["Prueba t de Student para Muestras Independientes", "Prueba t de Student Pareada (Medidas Repetidas)", "Prueba de Chi-cuadrado de Independencia", "ANOVA de una vía (Análisis de Varianza)", "Regresión Lineal Simple"]:
                    # All hypothesis tests have a p-value
                    p_val = res["p_value"]
                    stat_val = res.get("t_statistic") or res.get("chi2_statistic") or res.get("anova", {}).get("f_statistic") or res.get("f_statistic")
                    stat_name = "t" if "t_statistic" in res else ("Chi2" if "chi2_statistic" in res else "F")
                    
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
                        st.markdown(f"* **Ecuación estimada:** $Y = {res['intercept']:.4f} + {res['slope']:.4f} \cdot X$")
                        st.markdown(f"* **Coeficiente R-cuadrado ($R^2$):** `{res['r_squared']:.5f}` ({res['r_squared']*100:.2f}% de la varianza explicada)")
                        st.markdown(f"* **Coeficiente R de Pearson:** `{res['r_value']:.5f}`")
                        
                    elif selected_test == "ANOVA de una vía (Análisis de Varianza)":
                        st.markdown("**Resumen de Medias Grupales:**")
                        g_stats_df = pd.DataFrame(res["groups"])
                        g_stats_df.columns = ["Grupo", "Tamaño (n)", "Media", "Desv. Est."]
                        st.dataframe(g_stats_df, hide_index=True, use_container_width=True)

                st.markdown("---")
                st.markdown("##### 🔣 Fórmulas y Desglose Matemático Paso a Paso (LaTeX)")
                st.latex(res["latex"])
                
            with col_calc2:
                st.markdown("### 📈 Visualización del Análisis")
                
                # Interactive Plots based on test results
                if selected_test == "Estadísticos Descriptivos":
                    raw_data = inputs["data"]
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