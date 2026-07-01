import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import algebra
import calculo
import estadistica
import finanzas
import geometria
import ia
import itertools
import logica
import matplotlib.pyplot as plt
import numpy as np
import optimizacion
import os
import pandas as pd
import re
import streamlit as st
import sympy as sp

# ================= FUNCIONES DE LÓGICA MATEMÁTICA =================

def evaluar_expresion(formula_original, contexto):
    expr = formula_original

    # 1. Sustituir variables por valores
    for var, val in sorted(contexto.items(), key=lambda x: len(x[0]), reverse=True):
        expr = re.sub(rf'\b{var}\b', str(val), expr)

    # 2. Convertir operadores al estilo Python
    expr = re.sub(r'<->|↔', ' == ', expr)
    expr = re.sub(r'->|→', ' <=imp= ', expr)   # marcador temporal
    expr = re.sub(r'\^|&', ' and ', expr)
    expr = re.sub(r'\bv\b|\|', ' or ', expr)
    expr = re.sub(r'~|¬', ' not ', expr)

    # 3. Resolver implicaciones: A <=imp= B → (not (A) or (B))
    # Iterar hasta que no queden implicaciones
    max_iter = 20
    i = 0
    while '<=imp=' in expr and i < max_iter:
        expr = re.sub(
            r'((?:[^\s()][^=<>]*?|True|False))\s*<=imp=\s*((?:not\s+)?(?:[^\s()][^=<>]*?|True|False))',
            r'(not (\1) or (\2))',
            expr
        )
        i += 1

    try:
        return bool(eval(expr, {"__builtins__": None}, {"True": True, "False": False}))
    except Exception as e:
        raise ValueError(f"Error evaluando '{expr}': {e}")

def generar_tabla_verdad(formula_original):
    palabras_reservadas = {"AND", "OR", "NOT", "TRUE", "FALSE"}
    variables = sorted(set(re.findall(r'\b[A-Z]\b', formula_original)))
    variables = [v for v in variables if v not in palabras_reservadas]

    if not variables:
        raise ValueError("No se detectaron variables (deben ser letras mayúsculas: P, Q, R...).")

    combinaciones = list(itertools.product([True, False], repeat=len(variables)))
    tabla = []
    resultados = []

    for comb in combinaciones:
        contexto = dict(zip(variables, comb))
        try:
            resultado = evaluar_expresion(formula_original, contexto)
        except Exception as e:
            raise ValueError(f"Fórmula no válida: {e}")
        fila = {var: ("V" if val else "F") for var, val in contexto.items()}
        fila["Resultado"] = "V" if resultado else "F"
        tabla.append(fila)
        resultados.append(resultado)

    if all(resultados):
        clasificacion = "Tautología (Siempre Verdadera)"
    elif not any(resultados):
        clasificacion = "Contradicción (Siempre Falsa)"
    else:
        clasificacion = "Contingencia (Depende de las variables)"

    return {"Variables": variables, "Tabla": tabla, "Clasificacion": clasificacion}

# ================= CONFIGURACIÓN DE PÁGINA =================
st.set_page_config(
    page_title="Agent-Math: Agente Matemático Universitario",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= ESTILOS CSS PERSONALIZADOS =================
st.markdown("""
<style>
    /* Estilos globales */
    .main {
        background-color: #0E1117;
        color: #F8F9FA;
    }
    
    /* Efecto Glassmorphism en tarjetas */
    .math-card {
        background: rgba(30, 30, 36, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }
    
    /* Indicador del Agente */
    .agent-status-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #00F5D4;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px #00F5D4;
    }
    
    .agent-console {
        background-color: #14171E;
        border-left: 4px solid #9B5DE5;
        border-radius: 8px;
        padding: 18px;
        font-family: 'Courier New', Courier, monospace;
        margin-top: 15px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    
    /* Títulos con gradientes */
    .gradient-text {
        background: linear-gradient(135deg, #9B5DE5 0%, #00F5D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Personalización de botones */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #9B5DE5 0%, #00F5D4 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6em 1.5em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(155, 93, 229, 0.3) !important;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 245, 212, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR: CONTROL & API KEY =================
st.sidebar.markdown(
    '# 🧠 <span class="gradient-text">Agent-Math</span>',
    unsafe_allow_html=True
)
st.sidebar.caption("Panel de Control del Agente de Razonamiento Simbólico")

# Cargar API Key del entorno o .env
api_key_cargada = ia.cargar_api_key()

st.sidebar.subheader("🔑 Credenciales del Agente")
if api_key_cargada:
    st.sidebar.success("API Key cargada desde `.env` (Seguro)")
    user_api_key = api_key_cargada
else:
    user_api_key = st.sidebar.text_input(
        "Ingrese su Gemini API Key",
        type="password",
        help="Obtén una API Key en Google AI Studio"
    )

# Estado del agente
rama_seleccionada = st.sidebar.radio(
    "Selecciona la rama a trabajar:",
    [
        "🏠 Inicio",
        "🔢 Álgebra Universitaria",
        "📈 Cálculo Diferencial e Integral",
        "📊 Estadística y Probabilidad",
        "📐 Geometría y Trigonometría",
        "🎯 Optimización (Programación Lineal)",
        "💵 Matemática Financiera",
        "🧠 Lógica Matemática"
    ]
)



# ================= CABECERA DE LA PÁGINA PRINCIPAL =================
st.markdown(
    f'# 🧠 Agent-Math: <span class="gradient-text">{rama_seleccionada[2:]}</span>',
    unsafe_allow_html=True
)
st.write("---")

# ================= RENDERIZADO DE RAMAS MATEMÁTICAS (ORDENADO ALFABÉTICAMENTE) =================

# ----------------- 0. INICIO -----------------
if rama_seleccionada == "🏠 Inicio":
    st.markdown("""
    <div style="background: rgba(30, 30, 36, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 22px; margin-bottom: 20px; backdrop-filter: blur(4px);">
        <h3>👋 ¡Bienvenido a Agent-Math!</h3>
        <p>Tu asistente inteligente y agente de razonamiento para matemáticas universitarias. Este agente te permite realizar cálculos exactos de forma simbólica, visualizar funciones, analizar datos estadísticos y resolver problemas complejos paso a paso con el poder de la Inteligencia Artificial.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### 🛠️ Módulos de Razonamiento Disponibles:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="math-card">
            <h4>🔢 Álgebra Universitaria</h4>
            <p>Operaciones con matrices simbólicas (determinantes, RREF, inversas), sistemas de ecuaciones lineales y análisis/factorización de polinomios.</p>
        </div>
        <div class="math-card">
            <h4>📈 Cálculo Diferencial e Integral</h4>
            <p>Cálculo de derivadas, localización de puntos críticos, graficación de funciones y cálculo de integrales definidas e indefinidas con áreas sombreadas.</p>
        </div>
        <div class="math-card">
            <h4>📊 Estadística y Probabilidad</h4>
            <p>Cálculo de medidas descriptivas de centralización y dispersión, visualización de datos mediante histogramas o cajas, y modelado de distribuciones normal y binomial.</p>
        </div>
        <div class="math-card">
            <h4>📐 Geometría y Trigonometría</h4>
            <p>Cálculo de distancias y rectas en el plano cartesiano, resolución de triángulos por leyes de seno/coseno, y graficador interactivo de funciones trigonométricas.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="math-card">
            <h4>🎯 Optimización (Programación Lineal)</h4>
            <p>Resolución gráfica para dos variables, método Simplex para N variables, programación lineal entera y modelado de transporte y asignación.</p>
        </div>
        <div class="math-card">
            <h4>💵 Matemática Financiera</h4>
            <p>Simulador de tablas de amortización (cuota fija, constante, gradiente aritmético y geométrico) y evaluación de proyectos de inversión mediante VAN y TIR.</p>
        </div>
        <div class="math-card">
            <h4>🧠 Lógica Matemática</h4>
            <p>Generador interactivo de tablas de verdad proposicionales y verificador automático de leyes lógicas equivalentes.</p>
        </div>
        <div class="math-card">
            <h4>🤖 Consultas con IA</h4>
            <p>Utiliza el Agente de Razonamiento Gemini en cada área para resolver ejercicios detalladamente paso a paso con salida formateada en LaTeX.</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- 1. ÁLGEBRA UNIVERSITARIA -----------------
elif rama_seleccionada == "🔢 Álgebra Universitaria":
    tab_matrices, tab_sistemas, tab_polinomios = st.tabs([
        "🧮 Operaciones Matriciales",
        "🔗 Sistemas de Ecuaciones",
        "🧬 Polinomios y Raíces"
    ])

    with tab_matrices:
        st.subheader("Calculadora de Matrices Simbólicas y Exactas")
        st.write("Define las dimensiones e ingresa los elementos de la matriz. Puedes ingresar fracciones (ej: `1/2`, `3/4`) u otros términos.")

        cols_m = st.columns(2)
        filas = cols_m[0].number_input("Número de Filas (A)", min_value=1, max_value=4, value=3)
        columnas = cols_m[1].number_input("Número de Columnas (A)", min_value=1, max_value=4, value=3)

        st.write("### Elementos de la Matriz A:")
        datos_matriz = []
        for i in range(filas):
            col_inputs = st.columns(columnas)
            fila_vals = []
            for j in range(columnas):
                val = col_inputs[j].text_input(f"A[{i+1},{j+1}]", value=f"{i+j}", key=f"mat_A_{i}_{j}")
                fila_vals.append(val)
            datos_matriz.append(fila_vals)

        if st.button("Calcular Operaciones de la Matriz"):
            try:
                matriz_sympy = [[sp.sympify(x) for x in fila] for fila in datos_matriz]
                resultados = algebra.resolver_matriz_operaciones(matriz_sympy)

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write("#### Matriz Original $A$:")
                    st.latex(sp.latex(resultados["Original"]))
                    st.write("#### Forma Escalonada Reducida por Filas (RREF):")
                    st.latex(sp.latex(resultados["RREF"]))
                    st.write(f"Columnas pivote: `{resultados['Pivotes']}`")

                with col_res2:
                    st.write("#### Transpuesta $A^T$:")
                    st.latex(sp.latex(resultados["Transpuesta"]))
                    st.write("#### Determinante $|A|$:")
                    if isinstance(resultados["Determinante"], str):
                        st.info(resultados["Determinante"])
                    else:
                        st.latex(sp.latex(resultados["Determinante"]))

                    st.write("#### Inversa $A^{-1}$:")
                    if isinstance(resultados["Inversa"], str):
                        st.info(resultados["Inversa"])
                    else:
                        st.latex(sp.latex(resultados["Inversa"]))
            except Exception as e:
                st.error(f"Error procesando la matriz: {e}")

    with tab_sistemas:
        st.subheader("Resolvedor de Sistemas de Ecuaciones Lineales")
        st.write("Escribe el sistema de ecuaciones. Coloca una ecuación por línea (ej: `2*x + y - z = 8` o `2*x + y - z - 8`).")

        variables = st.text_input("Variables del sistema (separadas por comas)", value="x, y, z")
        eq_texto = st.text_area(
            "Ecuaciones del sistema:",
            value="2*x + y - z = 8\nx - y + z = 1\n3*x + 2*y + z = 13",
            height=120
        )

        if st.button("Resolver Sistema Lineal"):
            try:
                lista_eqs = eq_texto.split("\n")
                res = algebra.resolver_sistema_lineal(lista_eqs, variables)
                st.success("¡Sistema solucionado de forma exacta!")

                st.write("### Ecuaciones evaluadas:")
                for eq in res["Ecuaciones_Sympy"]:
                    st.latex(sp.latex(eq) + " = 0")

                st.write("### Solución Hallada:")
                sol = res["Soluciones"]
                if isinstance(sol, dict):
                    for var, val in sol.items():
                        st.latex(f"{sp.latex(var)} = {sp.latex(val)}")
                elif isinstance(sol, list):
                    if not sol:
                        st.warning("El sistema no tiene soluciones o es inconsistente.")
                    else:
                        for s_tuple in sol:
                            st.latex(sp.latex(s_tuple))
                else:
                    st.latex(sp.latex(sol))
            except Exception as e:
                st.error(f"Error resolviendo el sistema: {e}")

    with tab_polinomios:
        st.subheader("Análisis de Polinomios, Factorización y Raíces")
        poli_input = st.text_input("Ingresa un polinomio en x (ej: x**3 - 6*x**2 + 11*x - 6)", value="x**3 - 6*x**2 + 11*x - 6")

        if st.button("Analizar Polinomio"):
            try:
                res = algebra.resolver_polinomio(poli_input)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### Polinomio Original:")
                    st.latex(sp.latex(res["Original"]))
                    st.write("#### Factorización de Expresión:")
                    st.latex(sp.latex(res["Factorizacion"]))
                with col2:
                    st.write("#### Raíces de la Ecuación $P(x) = 0$:")
                    for idx, raiz in enumerate(res["Raices"]):
                        st.latex(f"x_{idx+1} = {sp.latex(raiz)}")
                    st.write("#### Primera Derivada $P'(x)$:")
                    st.latex(sp.latex(res["Derivada"]))
            except Exception as e:
                st.error(f"Error procesando el polinomio: {e}")

# ----------------- 2. CÁLCULO DIFERENCIAL E INTEGRAL -----------------
elif rama_seleccionada == "📈 Cálculo Diferencial e Integral":
    tab_derivar, tab_integrar = st.tabs([
        "📉 Graficador e Derivadas",
        "🔍 Integrales Definidas e Indefinidas"
    ])

    with tab_derivar:
        st.subheader("Análisis de Derivadas y Puntos Críticos")
        expr_calc = st.text_input("Ingresa la función f(x) (ej: x**3 - 3*x)", value="x**3 - 3*x")

        cols_lim = st.columns(2)
        x_min = cols_lim[0].number_input("Límite inferior gráfico X", value=-5.0)
        x_max = cols_lim[1].number_input("Límite superior gráfico X", value=5.0)

        if st.button("Derivar y Graficar"):
            try:
                res = calculo.analizar_funcion(expr_calc)
                st.success("¡Análisis analítico completado!")
                st.write("#### Expresión y Derivadas:")
                st.latex(f"f(x) = {sp.latex(res['Original'])}")
                st.latex(f"f'(x) = {sp.latex(res['Derivada_1'])}")
                st.latex(f"f''(x) = {sp.latex(res['Derivada_2'])}")

                st.write("#### Puntos Críticos ($f'(x) = 0$):")
                if isinstance(res["Puntos_Criticos"], list):
                    if not res["Puntos_Criticos"]:
                        st.info("No se hallaron puntos críticos reales.")
                    else:
                        for p in res["Puntos_Criticos"]:
                            y_val = res["Original"].subs(sp.Symbol('x'), p)
                            st.latex(f"({p:.4f}, {float(y_val):.4f})")
                else:
                    st.write(res["Puntos_Criticos"])

                fig = calculo.graficar_funcion_y_derivada(expr_calc, "x", (x_min, x_max))
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab_integrar:
        st.subheader("Calculadora de Integrales")
        expr_int = st.text_input("Ingresa la función f(x) a integrar (ej: x**2 * cos(x))", value="x**2")

        st.write("### Opciones de Integración:")
        tipo_int = st.radio("Tipo de Integral:", ["Indefinida", "Definida"])

        if tipo_int == "Definida":
            cols_lims = st.columns(2)
            a_int = cols_lims[0].number_input("Límite Inferior (a)", value=0.0)
            b_int = cols_lims[1].number_input("Límite Superior (b)", value=2.0)

        if st.button("Integrar Expresión"):
            try:
                if tipo_int == "Indefinida":
                    res = calculo.analizar_funcion(expr_int)
                    st.write("#### Integral Indefinida $\\int f(x) dx$:")
                    st.latex(sp.latex(res["Integral_Indefinida"]) + " + C")
                else:
                    res = calculo.calcular_integral_definida(expr_int, a_int, b_int)
                    st.success("¡Integral definida calculada!")
                    st.write("#### Valor Exacto:")
                    st.latex(sp.latex(res["Valor_Exacto"]))
                    st.write(f"#### Valor Numérico Aproximado: `{res['Valor_Numerico']:.6f}`")
                    st.pyplot(res["Fig"])
            except Exception as e:
                st.error(f"Error integrando: {e}")

# ----------------- 3. ESTADÍSTICA Y PROBABILIDAD -----------------
elif rama_seleccionada == "📊 Estadística y Probabilidad":
    tab_desc, tab_prob = st.tabs([
        "📊 Estadística Descriptiva",
        "🎲 Probabilidad y Distribuciones"
    ])
    
    with tab_desc:
        st.subheader("Análisis Descriptivo de Datos")
        texto_datos = st.text_area(
            "Ingresa los datos separados por comas (ej: 10, 15, 23, 12, 18, 10, 14, 20):", 
            "10, 15, 23, 12, 18, 10, 14, 20",
            key="input_texto_datos"
        )
        tipo_grafico = st.selectbox(
            "Elige el tipo de gráfico:", 
            ["Histograma", "Gráfico de Líneas", "Diagrama de Caja"],
            key="select_tipo_grafico"
        )
        
        if st.button("Analizar Datos", key="btn_analizar_datos"):
            try:
                datos = [float(x.strip()) for x in texto_datos.split(",") if x.strip() != ""]
                if not datos:
                    st.warning("Por favor, ingresa al menos un dato numérico.")
                else:
                    res = estadistica.descriptiva(datos)
                    st.success("¡Cálculo descriptivo completado!")
                    col_met1, col_met2, col_met3 = st.columns(3)

                    with col_met1:
                        st.markdown("**Centralización**")
                        st.write(f"• Cantidad de datos ($N$): `{res['Cantidad']}`")
                        st.write(f"• Media ($\\bar{{x}}$): `{res['Media']:.4f}`")
                        st.write(f"• Mediana: `{res['Mediana']}`")
                        st.write(f"• Moda: `{res['Moda']}`")

                    with col_met2:
                        st.markdown("**Dispersión**")
                        st.write(f"• Mínimo: `{res['Minimo']}`")
                        st.write(f"• Máximo: `{res['Maximo']}`")
                        st.write(f"• Rango: `{res['Rango']}`")
                        
                        if isinstance(res["Varianza"], float):
                            st.write(f"• Varianza ($s^2$): `{res['Varianza']:.4f}`")
                        else:
                            st.write(f"• Varianza: `{res['Varianza']}`")
                            
                        if isinstance(res["Desviacion_Estandar"], float):
                            st.write(f"• Desviación estándar ($s$): `{res['Desviacion_Estandar']:.4f}`")
                        else:
                            st.write(f"• Desviación estándar: `{res['Desviacion_Estandar']}`")
                            
                        if isinstance(res["Coeficiente_Variacion (%)"], float):
                            st.write(f"• Coef. Variación: `{res['Coeficiente_Variacion (%)']:.2f}%`")
                        else:
                            st.write(f"• Coef. Variación: `{res['Coeficiente_Variacion (%)']}`")

                    with col_met3:
                        st.markdown("**Cuartiles y Percentiles**")
                        st.write(f"• Primer Cuartil ($Q_1$): `{res['Q1']}`")
                        st.write(f"• Segundo Cuartil ($Q_2$): `{res['Q2']}`")
                        st.write(f"• Tercer Cuartil ($Q_3$): `{res['Q3']}`")
                        st.write(f"• Percentil 10 ($P_{{10}}$): `{res['P10']:.2f}`")
                        st.write(f"• Percentil 90 ($P_{{90}}$): `{res['P90']:.2f}`")

                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    if tipo_grafico == "Histograma":
                        ax.hist(datos, bins=6, color="#00F5D4", edgecolor="black", alpha=0.8)
                        ax.set_title("Histograma de Frecuencias", color="white")
                    elif tipo_grafico == "Gráfico de Líneas":
                        ax.plot(datos, marker="o", color="#9B5DE5", linewidth=2)
                        ax.set_title("Evolución de los Datos", color="white")
                    else:
                        ax.boxplot(datos, vert=False, patch_artist=True,
                                   boxprops=dict(facecolor="#00F5D4", color="white"),
                                   medianprops=dict(color="#9B5DE5", linewidth=2.5),
                                   whiskerprops=dict(color="white"),
                                   capprops=dict(color="white"))
                        ax.set_title("Gráfico de Caja y Bigotes", color="white") 
                
                    fig.patch.set_facecolor("#0E1117")
                    ax.set_facecolor("#1E1E24")
                    ax.tick_params(colors="white")
                    ax.grid(color="#333333", linestyle=":", alpha=0.5)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error calculando estadísticas descriptivas: {e}")

    with tab_prob:
        st.subheader("Modelado de Distribuciones de Probabilidad")
        tipo_dist = st.radio("Elige la distribución:", ["Normal Continua", "Binomial Discreta"], key="radio_tipo_dist")

        if tipo_dist == "Normal Continua":
            col_norm = st.columns(4)
            mu = col_norm[0].number_input("Media (mu)", value=0.0, key="norm_mu")
            sigma = col_norm[1].number_input("Desv. Estándar (sigma)", value=1.0, min_value=0.01, key="norm_sigma")
            x_val = col_norm[2].number_input("Valor X a evaluar", value=1.0, key="norm_x")
            op_norm = col_norm[3].selectbox("Operación:", ["<= ", ">= "], key="norm_op")

            if st.button("Calcular Probabilidad Normal", key="btn_calc_normal"):
                try:
                    res_norm = estadistica.analizar_distribucion_normal(mu, sigma, x_val, op_norm.strip())
                    st.success(f"Probabilidad Hallada: {res_norm['Probabilidad']:.6f}")
                    st.latex(f"{res_norm['Titulo_Formula']} = {res_norm['Probabilidad']:.6f}")
                    st.pyplot(res_norm["Fig"])
                except Exception as e:
                    st.error(f"Error en distribución normal: {e}")
                    
        elif tipo_dist == "Binomial Discreta":
            col_bin = st.columns(4)
            n_bin = col_bin[0].number_input("Ensayos (n)", value=10, min_value=1, step=1, key="bin_n")
            p_bin = col_bin[1].number_input("Prob. Éxito (p)", value=0.5, min_value=0.0, max_value=1.0, step=0.05, key="bin_p")
            k_bin = col_bin[2].number_input("Éxitos (k)", value=5, min_value=0, max_value=int(n_bin), step=1, key="bin_k")
            op_bin = col_bin[3].selectbox("Operación:", ["=", "<=", ">="], key="bin_op")
            
            if st.button("Calcular Probabilidad Binomial", key="btn_calc_binomial"):
                try:
                    res_bin = estadistica.analizar_distribucion_binomial(int(n_bin), p_bin, int(k_bin), op_bin)
                    st.success(f"Probabilidad Hallada: {res_bin['Probabilidad']:.6f}")
                    st.latex(f"{res_bin['Titulo_Formula']} = {res_bin['Probabilidad']:.6f}")
                    st.pyplot(res_bin["Fig"])
                except Exception as e:
                    st.error(f"Error en distribución binomial: {e}")

# ----------------- 4. GEOMETRÍA Y TRIGONOMETRÍA -----------------
elif rama_seleccionada == "📐 Geometría y Trigonometría":
    tab_analitica, tab_triangulos, tab_trig, tab_ondas = st.tabs([
        "📏 Geometría Analítica",
        "📐 Solucionador de Triángulos",
        "🌀 Círculo Unitario e Identidades",
        "📈 Ecuaciones y Ondas Trigonométricas"
    ])

    with tab_analitica:
        st.subheader("Geometría Analítica entre dos Puntos")
        cols_pt = st.columns(4)
        x1 = cols_pt[0].number_input("Punto A - X", value=1.0)
        y1 = cols_pt[1].number_input("Punto A - Y", value=2.0)
        x2 = cols_pt[2].number_input("Punto B - X", value=5.0)
        y2 = cols_pt[3].number_input("Punto B - Y", value=6.0)

        if st.button("Calcular Recta y Distancia"):
            try:
                res = geometria.distancia_y_recta(x1, y1, x2, y2)
                st.success("¡Geometría calculada!")
                st.write(f"• Distancia entre A y B: `{res['Distancia']:.4f}`")
                st.write(f"• Punto Medio: `{res['Punto_Medio']}`")
                st.write(f"• Pendiente ($m$): `{res['Pendiente']}`")
                st.write("• Ecuación general de la recta:")
                st.latex(res["Latex_Ecuacion"])
                st.pyplot(res["Fig"])
            except Exception as e:
                st.error(f"Error calculando geometría: {e}")

    with tab_triangulos:
        st.subheader("Solucionador de Triángulos (Leyes de Seno/Coseno)")
        st.write("Ingresa al menos 3 datos para resolver el triángulo (DDD, DAD, DDA). Deja en blanco los valores desconocidos.")

        cols_t1 = st.columns(3)
        a_lado = cols_t1[0].text_input("Lado a", value="5")
        b_lado = cols_t1[1].text_input("Lado b", value="6")
        c_lado = cols_t1[2].text_input("Lado c", value="7")

        cols_t2 = st.columns(3)
        A_ang = cols_t2[0].text_input("Ángulo A (°)", value="")
        B_ang = cols_t2[1].text_input("Ángulo B (°)", value="")
        C_ang = cols_t2[2].text_input("Ángulo C (°)", value="")

        if st.button("Resolver Triángulo"):
            try:
                a_f = float(a_lado) if a_lado.strip() else None
                b_f = float(b_lado) if b_lado.strip() else None
                c_f = float(c_lado) if c_lado.strip() else None
                A_f = float(A_ang) if A_ang.strip() else None
                B_f = float(B_ang) if B_ang.strip() else None
                C_f = float(C_ang) if C_ang.strip() else None

                res = geometria.resolver_triangulo(a_f, b_f, c_f, A_f, B_f, C_f)
                st.success("¡Triángulo resuelto con éxito!")

                col_res_t1, col_res_t2 = st.columns(2)
                with col_res_t1:
                    st.write("#### Lados:")
                    st.write(f"• Lado a = `{res['Lado_a']:.4f}`")
                    st.write(f"• Lado b = `{res['Lado_b']:.4f}`")
                    st.write(f"• Lado c = `{res['Lado_c']:.4f}`")
                    st.write("#### Dimensiones:")
                    st.write(f"• Área = `{res['Area']:.4f}`")
                    st.write(f"• Perímetro = `{(res['Lado_a'] + res['Lado_b'] + res['Lado_c']):.4f}`")
                with col_res_t2:
                    st.write("#### Ángulos:")
                    st.write(f"• Ángulo A = `{res['Angulo_A']:.2f}°`")
                    st.write(f"• Ángulo B = `{res['Angulo_B']:.2f}°`")
                    st.write(f"• Ángulo C = `{res['Angulo_C']:.2f}°`")

                st.pyplot(res["Fig"])
            except Exception as e:
                st.error(f"Error resolviendo triángulo: {e}")

    with tab_trig:
        st.subheader("Trigonometría Interactiva")
        col_trig1, col_trig2 = st.columns([1, 1])
        with col_trig1:
            st.write("Herramientas interactivas del círculo unitario y funciones trigonométricas.")
            ang = st.sidebar.slider("Ángulo (grados)", -360, 360, 30)
            rad = np.deg2rad(ang)
            st.write(f"sin({ang}°) = `{np.sin(rad):.4f}`")
            st.write(f"cos({ang}°) = `{np.cos(rad):.4f}`")
            st.write(f"tan({ang}°) = `{np.tan(rad):.4f}`" if np.cos(rad) != 0 else "tan indefinido (cos = 0)")

        with col_trig2:
            st.write("Representación mínima del círculo unitario (vista previa).")
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), color="#9B5DE5")
            ax.scatter([np.cos(rad)], [np.sin(rad)], color="#00F5D4")
            ax.set_aspect('equal', 'box')
            ax.axis('off')
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#1E1E24")
            st.pyplot(fig)

    with tab_ondas:
        st.subheader("Ecuaciones y Ondas Trigonométricas")
        st.write("Herramientas para analizar ecuaciones trigonométricas y ondas.")
        func_onda = st.text_input("Función trigonométrica (ej: sin(x) + 0.5*sin(2*x))", value="sin(x)")
        x_min = st.number_input("X Mínimo", value=-2 * np.pi)
        x_max = st.number_input("X Máximo", value=2 * np.pi)
        
        if st.button("Graficar Onda Trigonométrica"):
            try:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                xs = np.linspace(x_min, x_max, 400)
                x = sp.symbols('x')
                f = sp.lambdify(x, sp.sympify(func_onda), 'numpy')
                ys = f(xs)
                ax.plot(xs, ys, color="#00F5D4")
                ax.set_facecolor("#1E1E24")
                fig.patch.set_facecolor("#0E1117")
                ax.tick_params(colors="white")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error graficando la onda trigonométrica: {e}")


# ----------------- 5. LÓGICA MATEMÁTICA -----------------
elif rama_seleccionada == "🧠 Lógica Matemática":
    import itertools
    import re

    tab_manual, tab_leyes = st.tabs([
        "✏️ Generador de Tablas de Verdad",
        "🧠 Verificador de Leyes Lógicas"
    ])

    with tab_manual:
        st.subheader("Generador de Tablas de Verdad Proposicionales")
        st.write("Ingresa una proposición lógica utilizando letras mayúsculas como variables ($P$, $Q$, $R$, etc.).")
        st.write("Operadores soportados: `^` (AND), `v` (OR), `~` (NOT), `->` (Condicional), `<->` (Bicondicional).")

        formula_logica = st.text_input("Ingresa la proposición lógica:", value="(P -> Q) -> (~Q -> ~P)", key="manual_logic_input")

        if st.button("Generar Tabla de Verdad", key="btn_manual_truth"):
            formula_limpia = formula_logica.strip()
            
            if not formula_limpia:
                st.warning("⚠️ Por favor, escribe una proposición lógica en el cuadro de texto.")
            else:
                try:
                    # Detectar variables únicas en mayúsculas (P, Q, R, etc.)
                    vars_encontradas = sorted(list(set(re.findall(r'\b[A-Z]\b', formula_limpia))))
                    
                    if not vars_encontradas:
                        st.warning("⚠️ No se detectaron variables proposicionales válidas (letras mayúsculas como P, Q, R).")
                    else:
                        combos = list(itertools.product([True, False], repeat=len(vars_encontradas)))
                        filas = []
                        
                        # Función de evaluación analítica robusta sin errores de sintaxis por operadores parciales
                        def evaluar_expresion_limpia(formula, valores_variables):
                            # Copiar la fórmula de trabajo
                            temp_exp = formula
                            
                            # 1. Resolver Bicondicionales primero <-> por su operador booleano equivalente (==)
                            temp_exp = temp_exp.replace("<->", " == ")
                            
                            # 2. Reemplazar variables por strings "True" / "False" reales
                            for v, val in valores_variables.items():
                                temp_exp = re.sub(r'\b' + v + r'\b', str(val), temp_exp)
                            
                            # 3. Traducir operadores convencionales de texto a palabras clave de Python
                            temp_exp = temp_exp.replace("^", " and ")
                            temp_exp = temp_exp.replace("v", " or ")
                            temp_exp = temp_exp.replace("~", " not ")
                            
                            # 4. Resolver condicionales por precedencia matemática recursiva (A -> B  =>  not A or B)
                            # Esto asegura que estructuras complejas agrupadas se calculen sin fallar el árbol sintáctico
                            while "->" in temp_exp:
                                # Buscar la condicional más interna o emparejada
                                match = re.search(r'([a-zA-ZTrueFalsenot\s()]+?)\s*->\s*([a-zA-ZTrueFalsenot\s()]+)', temp_exp)
                                if match:
                                    antecedente = match.group(1)
                                    consecuente = match.group(2)
                                    reemplazo = f"(not ({antecedente}) or ({consecuente}))"
                                    temp_exp = temp_exp.replace(f"{antecedente}->{consecuente}", reemplazo, 1)
                                    # Fallback por si hay variaciones de espacios en el input
                                    temp_exp = temp_exp.replace(f"{antecedente} -> {consecuente}", reemplazo, 1)
                                else:
                                    # Ajuste de contingencia por si los caracteres adyacentes no machean el regex estándar
                                    temp_exp = temp_exp.replace("->", " <= ")
                                    break
                            
                            return bool(eval(temp_exp, {"__builtins__": None}, {}))

                        for combo in combos:
                            valores_dict = dict(zip(vars_encontradas, combo))
                            resultado = evaluar_expresion_limpia(formula_limpia, valores_dict)
                            
                            fila_legible = {v: 'V' if valores_dict[v] else 'F' for v in vars_encontradas}
                            fila_legible["Resultado"] = 'V' if resultado else 'F'
                            filas.append(fila_legible)
                        
                        df_tabla = pd.DataFrame(filas)
                        
                        # Clasificar la tabla resultante
                        resultados_unicos = df_tabla["Resultado"].unique()
                        if len(resultados_unicos) == 1 and resultados_unicos[0] == 'V':
                            st.success("#### Clasificación de la Fórmula: **Tautología (Siempre Válida) ✅**")
                        elif len(resultados_unicos) == 1 and resultados_unicos[0] == 'F':
                            st.error("#### Clasificación de la Fórmula: **Contradicción (Siempre Falsa) ❌**")
                        else:
                            st.warning("#### Clasificación de la Fórmula: **Contingencia (Variable) 📊**")
                        
                        st.table(df_tabla)
                except Exception as e:
                    st.error(f"Error analítico: Revisa el balance de tus paréntesis abiertos y cerrados. Detalle: {e}")

    with tab_leyes:
        st.write("### 🧠 Verificador de Leyes Lógicas Autocontenidas")
        
        LEYES_AUTOCONTENIDAS = {
            "1. Identidad": {"vars": ["P"], "sym": "(P ↔ P)", "func": lambda P: P == P},
            "2. Doble negación": {"vars": ["P"], "sym": "P ↔ ¬(¬P)", "func": lambda P: P == (not (not P))},
            "3. Tercero excluido": {"vars": ["P"], "sym": "P ∨ ¬P", "func": lambda P: P or (not P)},
            "4. No contradicción": {"vars": ["P"], "sym": "¬(P ∧ ¬P)", "func": lambda P: not (P and (not P))},
            "5. Conmutativa AND": {"vars": ["P", "Q"], "sym": "(P ∧ Q) ↔ (Q ∧ P)", "func": lambda P, Q: (P and Q) == (Q and P)},
            "6. Conmutativa OR": {"vars": ["P", "Q"], "sym": "(P ∨ Q) ↔ (Q ∨ P)", "func": lambda P, Q: (P or Q) == (Q or P)},
            "7. Asociativa AND": {"vars": ["P", "Q", "R"], "sym": "((P ∧ Q) ∧ R) ↔ (P ∧ (Q ∧ R))", "func": lambda P, Q, R: ((P and Q) and R) == (P and (Q and R))},
            "8. Asociativa OR": {"vars": ["P", "Q", "R"], "sym": "((P ∨ Q) ∨ R) ↔ (P ∨ (Q ∨ R))", "func": lambda P, Q, R: ((P or Q) or R) == (P or (Q or R))},
            "9. Distributiva 1": {"vars": ["P", "Q", "R"], "sym": "[P ∧ (Q ∨ R)] ↔ [(P ∧ Q) ∨ (P ∧ R)]", "func": lambda P, Q, R: (P and (Q or R)) == ((P and Q) or (P and R))},
            "10. Distributiva 2": {"vars": ["P", "Q", "R"], "sym": "[P ∨ (Q ∧ R)] ↔ [(P ∨ Q) ∧ (P ∨ R)]", "func": lambda P, Q, R: (P or (Q and R)) == ((P or Q) and (P or R))},
            "11. De Morgan 1": {"vars": ["P", "Q"], "sym": "¬(P ∧ Q) ↔ (¬P ∨ ¬Q)", "func": lambda P, Q: (not (P and Q)) == ((not P) or (not Q))},
            "12. De Morgan 2": {"vars": ["P", "Q"], "sym": "¬(P ∨ Q) ↔ (¬P ∧ ¬Q)", "func": lambda P, Q: (not (P or Q)) == ((not P) and (not Q))},
            "13. Absorción 1": {"vars": ["P", "Q"], "sym": "[P ∨ (P ∧ Q)] ↔ P", "func": lambda P, Q: (P or (P and Q)) == P},
            "14. Absorción 2": {"vars": ["P", "Q"], "sym": "[P ∧ (P ∨ Q)] ↔ P", "func": lambda P, Q: (P and (P or Q)) == P},
            "15. Implicación": {"vars": ["P", "Q"], "sym": "(P → Q) ↔ (¬P ∨ Q)", "func": lambda P, Q: (not P or Q) == (not P or Q)},
            "16. Contraposición": {"vars": ["P", "Q"], "sym": "(P → Q) ↔ (¬Q → ¬P)", "func": lambda P, Q: (not P or Q) == (not Q or not P)},
            "17. Bicondicional": {"vars": ["P", "Q"], "sym": "(P ↔ Q) ↔ [(P → Q) ∧ (Q → P)]", "func": lambda P, Q: (P == Q) == ((not P or Q) and (not Q or P))},
        }

        ley_seleccionada = st.selectbox(
            "Selecciona una ley lógica para evaluar su tabla de verdad:",
            options=list(LEYES_AUTOCONTENIDAS.keys()),
            key="ley_logica_autocontenida"
        )

        if ley_seleccionada:
            ley_info = LEYES_AUTOCONTENIDAS[ley_seleccionada]
            st.info(f"**Fórmula formal:** {ley_info['sym']}")
            
            if st.button("Evaluar Ley Seleccionada", key="btn_evaluar_ley_autocontenida"):
                try:
                    vars_list = ley_info["vars"]
                    func = ley_info["func"]
                    combos = list(itertools.product([True, False], repeat=len(vars_list)))
                    
                    filas = []
                    for combo in combos:
                        valores_dict = dict(zip(vars_list, combo))
                        resultado = func(*combo)
                        
                        fila_legible = {v: 'V' if valores_dict[v] else 'F' for v in vars_list}
                        fila_legible["Resultado de la Ley"] = 'V' if resultado else 'F'
                        filas.append(fila_legible)
                    
                    df_tabla = pd.DataFrame(filas)
                    st.dataframe(df_tabla, use_container_width=True)
                    
                    if all(df_tabla["Resultado de la Ley"] == 'V'):
                        st.success("✅ ¡Se demuestra matemáticamente que es una **Tautología** (Ley Válida omnipresente)!")
                except Exception as e:
                    st.error(f"Error generando tabla analítica: {e}")



# ----------------- 6. MATEMÁTICA FINANCIERA -----------------
elif rama_seleccionada == "💵 Matemática Financiera":
    import finanzas

    tab_amort, tab_van = st.tabs([
        "📉 Amortización de Préstamos",
        "📊 Evaluación de Proyectos (VAN y TIR)"
    ])

    with tab_amort:
        st.subheader("Generador de Tablas de Amortización Dinámicas")

        cols_fin = st.columns(3)
        capital = cols_fin[0].number_input("Capital Prestado ($)", min_value=100.0, value=10000.0, step=500.0)
        tasa_nominal = cols_fin[1].number_input("Tasa Nominal Anual (%)", min_value=0.0, value=12.0, step=0.5)
        meses_plazo = cols_fin[2].number_input("Plazo en Meses", min_value=1, value=12, step=1)

        st.write("---")

        sistema_amort = st.radio(
            "Seleccione el Sistema de Amortización:",
            [
                "Sistema de Cuota Fija", 
                "Sistema de Amortización Constante (Cuotas Decrecientes)",
                "Sistema con Gradiente Aritmético (Variación Fija en $)",
                "Sistema con Gradiente Geométrico (Variación Porcentual %)"
            ]
        )

        gradiente_valor = 0.0
        es_creciente = True

        if "Gradiente" in sistema_amort:
            st.info("💡 **Configuración del Gradiente:** Defina la variación que sufrirá la cuota mes a mes.")
            col_g1, col_g2 = st.columns(2)
            
            if "Aritmético" in sistema_amort:
                gradiente_valor = col_g1.number_input("Valor del Gradiente ($ de variación mensual)", min_value=0.0, value=50.0, step=5.0)
            else:
                gradiente_valor = col_g1.number_input("Porcentaje de Gradiente (% de variación mensual)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
                
            sentido = col_g2.radio("Sentido de la variación:", ["Creciente (Las cuotas aumentan)", "Decreciente (Las cuotas disminuyen)"])
            es_creciente = "Creciente" in sentido

        st.write("---")

        if st.button("Calcular y Generar Tabla", key="btn_calcular_amort_nativa"):
            try:
                cuota_info_str = ""

                if "Cuota Fija" in sistema_amort:
                    tabla_datos, cuota_fija = finanzas.amortizacion_cuota_fija(capital, tasa_nominal, meses_plazo)
                    cuota_info_str = f"Cuota Mensual Fija: **${cuota_fija:.2f}**"

                elif "Amortización Constante" in sistema_amort:
                    tabla_datos = finanzas.amortizacion_cuota_decreciente(capital, tasa_nominal, meses_plazo)
                    cuota_info_str = f"Amortización Constante a Capital"

                elif "Aritmético" in sistema_amort:
                    tabla_datos, a1 = finanzas.amortizacion_gradiente_aritmetico(capital, tasa_nominal, meses_plazo, gradiente_valor, es_creciente)
                    cuota_info_str = f"Primera Cuota Base (A1): **${a1:.2f}** | Variación: **{'+' if es_creciente else '-'}${gradiente_valor}** mensual"

                elif "Geométrico" in sistema_amort:
                    tabla_datos, a1 = finanzas.amortizacion_gradiente_geometrico(capital, tasa_nominal, meses_plazo, gradiente_valor, es_creciente)
                    cuota_info_str = f"Primera Cuota Base (A1): **${a1:.2f}** | Variación: **{'+' if es_creciente else '-'}{gradiente_valor}%** mensual"

                st.success(f"📊 **Plan de Pagos Estructurado** | {cuota_info_str}")
                df_amort = pd.DataFrame(tabla_datos)
                st.dataframe(df_amort, use_container_width=True)

                st.write("### 📈 Curva de Amortización del Saldo")
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(df_amort["Mes"], df_amort["Saldo_Restante"], marker="o", color="#00F5D4", linewidth=2.5)
                ax.set_title("Evolución del Saldo Restante del Crédito", color="white", fontsize=10)
                ax.set_xlabel("Meses", color="white")
                ax.set_ylabel("Saldo ($)", color="white")
                ax.grid(True, linestyle=":", alpha=0.4, color="#333333")
                ax.set_facecolor("#1E1E24")
                fig.patch.set_facecolor("#0E1117")
                ax.tick_params(colors="white")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error al procesar el módulo financiero: {e}")

    with tab_van:
        st.subheader("Evaluación de Proyectos de Inversión (VAN y TIR)")
        st.write("Ingresa la inversión inicial, los flujos de caja netos esperados por periodo y la tasa de descuento anual.")

        col_van1, col_van2 = st.columns(2)
        inversion_inicial = col_van1.number_input("Inversión Inicial ($)", min_value=0.0, value=10000.0, step=500.0)
        tasa_descuento = col_van2.number_input("Tasa de Descuento Anual (%)", min_value=0.0, value=10.0, step=0.5)

        flujos_texto = st.text_area(
            "Flujos de Caja Netos por periodo (separados por comas, en orden: periodo 1, 2, 3...):",
            value="3000, 4000, 4000, 3500",
            height=80
        )

        if st.button("Calcular VAN y TIR", key="btn_calcular_van_nativo"):
            try:
                flujos = [float(x.strip()) for x in flujos_texto.split(",") if x.strip() != ""]
                if not flujos:
                    st.warning("Debes ingresar al menos un flujo de caja.")
                else:
                    # Llamamos al cajón matemático unificado de finanzas.py
                    van, tir, detalle_flujos = finanzas.evaluar_proyecto(inversion_inicial, flujos, tasa_descuento)

                    st.success("¡Evaluación del proyecto completada!")
                    col_res_van1, col_res_van2 = st.columns(2)
                    with col_res_van1:
                        st.write(f"#### VAN (Valor Actual Neto): `${van:.2f}`")
                        if tir is not None:
                            st.write(f"#### TIR (Tasa Interna de Retorno): `{tir:.2f}%`")
                        else:
                            st.info("TIR no disponible o fuera de rango matemático estándar.")

                    with col_res_van2:
                        if van > 0:
                            st.success("Decisión: ¡Proyecto Viable! El rendimiento supera la tasa exigida.")
                        elif van < 0:
                            st.error("Decisión: Proyecto No Viable. No se recupera la tasa de descuento.")
                        else:
                            st.info("Decisión: Indiferente. El proyecto está en el punto de equilibrio.")

                    df_van = pd.DataFrame(detalle_flujos)
                    st.dataframe(df_van, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    ax.bar(df_van["Periodo"], df_van["Valor_Presente"], color="#9B5DE5", alpha=0.85)
                    ax.set_title("Valor Presente de los Flujos de Caja por Periodo", color="white", fontsize=10)
                    ax.set_xlabel("Periodo", color="white")
                    ax.set_ylabel("Valor Presente ($)", color="white")
                    ax.grid(True, linestyle=":", alpha=0.4, color="#333333")
                    ax.set_facecolor("#1E1E24")
                    fig.patch.set_facecolor("#0E1117")
                    ax.tick_params(colors="white")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error evaluando el proyecto: {e}")


# ----------------- 7. OPTIMIZACIÓN (P. LINEAL) -----------------
elif rama_seleccionada == "🎯 Optimización (Programación Lineal)":
    from optimizacion_app import mostrar_optimizacion
    mostrar_optimizacion()


# ================= CONSOLA DE RAZONAMIENTO DEL AGENTE (GEMINI) =================
st.write("---")
with st.expander("🤖 Consola del Agente de Razonamiento Gemini", expanded=False):
    st.markdown("### Haz una consulta al Agente sobre esta área de estudio:")
    
    tema = "matemáticas generales" if rama_seleccionada == "🏠 Inicio" else rama_seleccionada[2:]
    st.write(f"El agente analizará tu ejercicio en el contexto de: **{tema}**")
    
    pregunta_usuario = st.text_area(
        "Describe tu ejercicio o problema matemático aquí:", 
        placeholder="Ejemplo: Resuelve el sistema de ecuaciones 2x + y = 5, x - y = 1 paso a paso.",
        key="pregunta_agente"
    )
    
    if st.button("Consultar al Agente", key="btn_consultar_agente"):
        if not pregunta_usuario.strip():
            st.warning("Por favor, escribe una pregunta.")
        elif not user_api_key:
            st.error("Por favor, ingresa tu Gemini API Key en la barra lateral para activar el agente.")
        else:
            with st.spinner("El Agente está pensando..."):
                try:
                    respuesta = ia.responder(pregunta_usuario, api_key=user_api_key, tema=tema)
                    st.markdown("#### 💻 Respuesta del Agente:")
                    st.markdown(f'<div class="agent-console">', unsafe_allow_html=True)
                    st.markdown(respuesta)
                    st.markdown(f'</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error al consultar el agente: {e}")