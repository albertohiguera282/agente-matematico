"""
Módulo: Calculadora Científica
--------------------------------
Calculadora de funciones operacionales científicas con "cajón amplio"
(display extendido + historial tipo cinta de papel) para reforzar al
Agente Matemático (Agent-Math) como apoyo de ejemplos analíticos rápidos.

Expone una única función pública: mostrar_calculadora()
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import ia


# ─────────────────────────────────────────────
# NAMESPACE SEGURO DE EVALUACIÓN (sin symbolic, numérico puro)
# ─────────────────────────────────────────────
def _namespace(modo_angulo: str):
    """Construye el diccionario de funciones disponibles para eval(),
    adaptando las funciones trigonométricas al modo Grados/Radianes."""

    def sin_(x):
        return math.sin(math.radians(x)) if modo_angulo == "Grados" else math.sin(x)

    def cos_(x):
        return math.cos(math.radians(x)) if modo_angulo == "Grados" else math.cos(x)

    def tan_(x):
        return math.tan(math.radians(x)) if modo_angulo == "Grados" else math.tan(x)

    def asin_(x):
        r = math.asin(x)
        return math.degrees(r) if modo_angulo == "Grados" else r

    def acos_(x):
        r = math.acos(x)
        return math.degrees(r) if modo_angulo == "Grados" else r

    def atan_(x):
        r = math.atan(x)
        return math.degrees(r) if modo_angulo == "Grados" else r

    return {
        "sin": sin_, "cos": cos_, "tan": tan_,
        "asin": asin_, "acos": acos_, "atan": atan_,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
        "log": math.log10, "ln": math.log, "log2": math.log2,
        "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
        "exp": math.exp,
        "pi": math.pi, "e": math.e,
        "factorial": math.factorial,
        "comb": math.comb, "perm": math.perm,
        "abs": abs, "pow": pow, "round": round,
        "__builtins__": {},
    }


def _evaluar(expr_str: str, modo_angulo: str):
    """Evalúa una expresión numérica de forma segura (sin acceso a builtins)."""
    if not expr_str or not expr_str.strip():
        raise ValueError("La expresión está vacía.")
    limpio = expr_str.replace("^", "**").replace("√", "sqrt").replace(",", ".")
    ns = _namespace(modo_angulo)
    resultado = eval(limpio, {"__builtins__": {}}, ns)  # noqa: S307 - namespace restringido
    return resultado


# ─────────────────────────────────────────────
# GRAFICACIÓN DE FUNCIONES f(x)
# ─────────────────────────────────────────────
def _namespace_np(modo_angulo: str):
    """Igual que _namespace pero vectorizado con numpy, para graficar f(x)."""

    def sin_(x):
        return np.sin(np.radians(x)) if modo_angulo == "Grados" else np.sin(x)

    def cos_(x):
        return np.cos(np.radians(x)) if modo_angulo == "Grados" else np.cos(x)

    def tan_(x):
        return np.tan(np.radians(x)) if modo_angulo == "Grados" else np.tan(x)

    return {
        "sin": sin_, "cos": cos_, "tan": tan_,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "log": np.log10, "ln": np.log, "log2": np.log2,
        "sqrt": np.sqrt, "cbrt": np.cbrt, "exp": np.exp,
        "pi": np.pi, "e": np.e,
        "abs": np.abs, "pow": np.power,
        "__builtins__": {},
    }


def _graficar_funcion(expr_str: str, modo_angulo: str, x_min: float, x_max: float):
    """Genera la figura matplotlib de f(x) en el rango [x_min, x_max]."""
    if not expr_str or not expr_str.strip():
        raise ValueError("Ingresa una función de x, por ejemplo: sin(x)+x^2")
    if x_min >= x_max:
        raise ValueError("El límite inferior debe ser menor que el superior.")

    limpio = expr_str.replace("^", "**").replace("√", "sqrt")
    x = np.linspace(x_min, x_max, 800)
    ns = _namespace_np(modo_angulo)
    ns["x"] = x

    with np.errstate(divide="ignore", invalid="ignore"):
        y = eval(limpio, {"__builtins__": {}}, ns)  # noqa: S307 - namespace restringido

    if np.isscalar(y):
        y = np.full_like(x, float(y))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x, y, color="#00F5D4", lw=2.2, label=f"$f(x) = {expr_str}$")
    ax.axhline(0, color="white", lw=1)
    ax.axvline(0, color="white", lw=1)
    ax.grid(color="#333333", linestyle=":", alpha=0.5)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("f(x)", color="white")
    ax.set_title(f"Gráfica de f(x) = {expr_str}", color="white")
    ax.legend(facecolor="#1e1e24", edgecolor="#9B5DE5", labelcolor="white", fontsize=9)
    ax.set_facecolor("#1E1E24")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#333333")

    return fig


# ─────────────────────────────────────────────
# CALLBACKS DE ESTADO (botones tipo calculadora física)
# ─────────────────────────────────────────────
def _init_state():
    if "calc_expr" not in st.session_state:
        st.session_state.calc_expr = ""
    if "calc_memoria" not in st.session_state:
        st.session_state.calc_memoria = 0.0
    if "calc_historial" not in st.session_state:
        st.session_state.calc_historial = []  # lista de (expresion, resultado)
    if "calc_ultimo_resultado" not in st.session_state:
        st.session_state.calc_ultimo_resultado = None


def _insertar(token: str):
    st.session_state.calc_expr += token


def _limpiar():
    st.session_state.calc_expr = ""


def _borrar_ultimo():
    st.session_state.calc_expr = st.session_state.calc_expr[:-1]


def _calcular(modo_angulo: str):
    try:
        resultado = _evaluar(st.session_state.calc_expr, modo_angulo)
        st.session_state.calc_ultimo_resultado = resultado
        st.session_state.calc_historial.insert(
            0, (st.session_state.calc_expr, resultado)
        )
        st.session_state.calc_historial = st.session_state.calc_historial[:30]
        st.session_state.calc_expr = str(resultado)
    except Exception as e:
        st.session_state.calc_ultimo_resultado = f"Error: {e}"


def _mem_mas():
    try:
        st.session_state.calc_memoria += float(st.session_state.calc_ultimo_resultado)
    except (TypeError, ValueError):
        pass


def _mem_menos():
    try:
        st.session_state.calc_memoria -= float(st.session_state.calc_ultimo_resultado)
    except (TypeError, ValueError):
        pass


def _mem_recuperar():
    st.session_state.calc_expr += str(st.session_state.calc_memoria)


def _mem_limpiar():
    st.session_state.calc_memoria = 0.0


# ─────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────
def mostrar_calculadora():
    _init_state()

    st.markdown("### 🧮 Calculadora Científica")
    st.caption(
        "Funciones operacionales completas + cajón amplio de historial, "
        "como refuerzo de cálculo rápido para el Agente Matemático."
    )

    tab_calc, tab_graf = st.tabs(["🧮 Calculadora", "📈 Graficador de Funciones"])

    with tab_calc:
        _tab_calculadora()

    with tab_graf:
        _tab_graficador()


def _tab_calculadora():
    col_calc, col_hist = st.columns([1.3, 1])

    # ---------------- COLUMNA CALCULADORA ----------------
    with col_calc:
        modo_angulo = st.radio(
            "Modo angular", ["Grados", "Radianes"], horizontal=True, key="calc_modo_angulo"
        )

        st.text_input(
            "Display",
            key="calc_expr",
            label_visibility="collapsed",
            placeholder="Escribe o usa los botones (ej: sin(30)+sqrt(16))",
        )

        if isinstance(st.session_state.calc_ultimo_resultado, (int, float)):
            st.markdown(f"**= {st.session_state.calc_ultimo_resultado}**")
        elif isinstance(st.session_state.calc_ultimo_resultado, str):
            st.error(st.session_state.calc_ultimo_resultado)

        # Fila de memoria
        m1, m2, m3, m4 = st.columns(4)
        m1.button("MC", use_container_width=True, on_click=_mem_limpiar)
        m2.button("MR", use_container_width=True, on_click=_mem_recuperar)
        m3.button("M+", use_container_width=True, on_click=_mem_mas)
        m4.button("M-", use_container_width=True, on_click=_mem_menos)

        # Fila de funciones científicas
        filas_funciones = [
            ["sin(", "cos(", "tan(", "π", "e"],
            ["asin(", "acos(", "atan(", "ln(", "log("],
            ["sqrt(", "cbrt(", "^", "exp(", "!"],
            ["(", ")", "comb(", "perm(", "abs("],
        ]
        simbolo_a_token = {"π": "pi", "!": "factorial("}
        for fila in filas_funciones:
            cols = st.columns(len(fila))
            for c, simbolo in zip(cols, fila):
                token = simbolo_a_token.get(simbolo, simbolo)
                c.button(simbolo, key=f"btn_{simbolo}", use_container_width=True,
                         on_click=_insertar, args=(token,))

        # Teclado numérico + operadores
        filas_numeros = [
            ["7", "8", "9", "/"],
            ["4", "5", "6", "*"],
            ["1", "2", "3", "-"],
            ["0", ".", "%", "+"],
        ]
        for fila in filas_numeros:
            cols = st.columns(4)
            for c, simbolo in zip(cols, fila):
                c.button(simbolo, key=f"btn_num_{simbolo}", use_container_width=True,
                         on_click=_insertar, args=(simbolo,))

        b1, b2, b3 = st.columns(3)
        b1.button("⌫ Borrar", use_container_width=True, on_click=_borrar_ultimo)
        b2.button("C Limpiar", use_container_width=True, on_click=_limpiar)
        b3.button("= Calcular", type="primary", use_container_width=True,
                  on_click=_calcular, args=(modo_angulo,))

    # ---------------- COLUMNA "CAJÓN AMPLIO" (HISTORIAL) ----------------
    with col_hist:
        st.markdown("#### 🧾 Historial (cajón amplio)")
        if not st.session_state.calc_historial:
            st.info("Aún no hay cálculos. El historial se llenará aquí.")
        else:
            for i, (expr, resultado) in enumerate(st.session_state.calc_historial):
                with st.container(border=True):
                    st.markdown(f"`{expr}` → **{resultado}**")
                    if st.button("🤖 Explicar con IA", key=f"explicar_{i}"):
                        api_key = ia.cargar_api_key() or st.session_state.get("user_api_key")
                        if not api_key:
                            st.warning(
                                "Ingresa tu Gemini API Key en la barra lateral para "
                                "activar la explicación del agente."
                            )
                        else:
                            with st.spinner("El Agente está analizando el cálculo..."):
                                try:
                                    pregunta = (
                                        f"Explica paso a paso, con fundamento teórico, "
                                        f"cómo se resuelve la operación: {expr} "
                                        f"(resultado obtenido: {resultado})."
                                    )
                                    respuesta = ia.responder(
                                        pregunta, api_key=api_key,
                                        tema="Calculadora Científica"
                                    )
                                    st.markdown(respuesta)
                                except Exception as e:
                                    st.error(f"Error al consultar el agente: {e}")

            if st.button("🗑️ Vaciar historial"):
                st.session_state.calc_historial = []
                st.rerun()


def _tab_graficador():
    st.caption(
        "Grafica cualquier función f(x) usando las mismas funciones científicas "
        "de la calculadora (sin, cos, log, sqrt, exp, ^, etc.)."
    )

    modo_angulo_g = st.radio(
        "Modo angular", ["Grados", "Radianes"], horizontal=True, key="graf_modo_angulo"
    )

    expr_funcion = st.text_input(
        "f(x) =",
        value="sin(x)",
        placeholder="Ejemplo: x^2 - 3*x + 2   |   sin(x)+cos(x)   |   ln(x)",
        key="graf_expr",
    )

    col_min, col_max = st.columns(2)
    x_min = col_min.number_input("x mínimo", value=-10.0, step=1.0, key="graf_xmin")
    x_max = col_max.number_input("x máximo", value=10.0, step=1.0, key="graf_xmax")

    if st.button("📈 Graficar función", type="primary", key="btn_graficar"):
        try:
            fig = _graficar_funcion(expr_funcion, modo_angulo_g, x_min, x_max)
            st.pyplot(fig)

            if st.button("🤖 Explicar esta función con IA", key="btn_explicar_grafica"):
                api_key = ia.cargar_api_key() or st.session_state.get("user_api_key")
                if not api_key:
                    st.warning(
                        "Ingresa tu Gemini API Key en la barra lateral para "
                        "activar la explicación del agente."
                    )
                else:
                    with st.spinner("El Agente está analizando la función..."):
                        try:
                            pregunta = (
                                f"Analiza la función f(x) = {expr_funcion} en el "
                                f"intervalo [{x_min}, {x_max}]: describe su dominio, "
                                f"comportamiento, posibles ceros, y puntos relevantes."
                            )
                            respuesta = ia.responder(
                                pregunta, api_key=api_key,
                                tema="Calculadora Científica - Graficador"
                            )
                            st.markdown(respuesta)
                        except Exception as e:
                            st.error(f"Error al consultar el agente: {e}")
        except Exception as e:
            st.error(f"No se pudo graficar la función: {e}")
