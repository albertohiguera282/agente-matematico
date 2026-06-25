import streamlit as st
import numpy as np
import pandas as pd
from optimizacion import (
    resolver_programacion_lineal_grafico,
    resolver_programacion_lineal_simplex,
    resolver_programacion_lineal_entera,
    analizar_sensibilidad,
    resolver_transporte,
    resolver_asignacion,
)

def _parse_fo(text, n):
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != n:
        raise ValueError(f"Se esperaban {n} coeficientes, se recibieron {len(vals)}.")
    return vals

def mostrar_optimizacion():
    st.title("📐 Optimización — Programación Lineal y Extensiones")
    tabs = st.tabs([
        "1️⃣ Gráfico 2D",
        "2️⃣ Simplex N-Variables",
        "3️⃣ PL Entera",
        "4️⃣ Sensibilidad",
        "5️⃣ Transporte",
        "6️⃣ Asignación",
    ])

    with tabs[0]:
        st.subheader("Método Gráfico (2 variables)")
        col1, col2 = st.columns(2)
        with col1:
            c1 = st.number_input("Coeficiente c₁", value=5.0, key="g_c1")
            c2 = st.number_input("Coeficiente c₂", value=4.0, key="g_c2")
            tipo = st.selectbox("Tipo", ["max", "min"], key="g_tipo")
        with col2:
            n_rest = st.number_input("Número de restricciones", 1, 8, 2, step=1, key="g_nrest")
        restricciones = []
        cols_r = st.columns(3)
        for i in range(int(n_rest)):
            with cols_r[0]:
                a = st.number_input(f"a{i+1}", value=1.0, key=f"ga{i}")
            with cols_r[1]:
                b = st.number_input(f"b{i+1}", value=1.0, key=f"gb{i}")
            with cols_r[2]:
                rhs = st.number_input(f"rhs{i+1}", value=6.0, key=f"gr{i}")
            restricciones.append((a, b, rhs))
        if st.button("Resolver gráficamente", key="btn_graf"):
            try:
                res = resolver_programacion_lineal_grafico((c1, c2), restricciones, tipo)
                x_opt, y_opt = res["Punto_Optimo"]
                st.success(f"x₁ = {x_opt:.4f}, x₂ = {y_opt:.4f} | Z = {res['Z_Optimo']:.4f}")
                st.pyplot(res["Fig"])
                st.dataframe(pd.DataFrame(res["Vertices_Evaluados"]), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tabs[1]:
        st.subheader("Método Simplex — N Variables")
        tipo_s = st.selectbox("Tipo", ["max", "min"], key="s_tipo")
        n_vars_s = st.number_input("Variables", 2, 10, 3, key="s_nv")
        n_rest_s = st.number_input("Restricciones", 1, 15, 3, key="s_nr")
        fo_input = st.text_input("Función objetivo (coefs separados por coma)", value="40,30,50", key="s_fo")
        rest_inputs, rhs_inputs = [], []
        for i in range(int(n_rest_s)):
            c_col, r_col = st.columns([3, 1])
            with c_col:
                ri = st.text_input(f"Coefs R{i+1}", value=",".join(["1"]*int(n_vars_s)), key=f"sr_coef{i}")
            with r_col:
                bi = st.number_input(f"RHS {i+1}", value=100.0, key=f"sr_rhs{i}")
            rest_inputs.append(ri); rhs_inputs.append(bi)
        if st.button("Resolver Simplex", key="btn_simplex"):
            try:
                fo = _parse_fo(fo_input, int(n_vars_s))
                A = [_parse_fo(r, int(n_vars_s)) for r in rest_inputs]
                res = resolver_programacion_lineal_simplex(fo, A, list(rhs_inputs), tipo_s)
                if res["Estado"] == "Óptimo":
                    st.success(f"Z = {res['Valor_Optimo']}")
                    st.dataframe(pd.DataFrame(res["Variables"].items(), columns=["Variable", "Valor"]), use_container_width=True)
                else:
                    st.warning(res.get("Mensaje", ""))
            except Exception as e:
                st.error(f"Error: {e}")

    with tabs[2]:
        st.subheader("Programación Lineal Entera / Mixta")
        tipo_e = st.selectbox("Tipo", ["max", "min"], key="e_tipo")
        n_vars_e = st.number_input("Variables", 2, 10, 3, key="e_nv")
        n_rest_e = st.number_input("Restricciones", 1, 15, 2, key="e_nr")
        todas_enteras = st.checkbox("Todas las variables enteras", value=True, key="e_all")
        fo_e = st.text_input("Función objetivo", value="5,8,3", key="e_fo")
        rest_e, rhs_e = [], []
        for i in range(int(n_rest_e)):
            cc, rc = st.columns([3, 1])
            with cc:
                ri = st.text_input(f"Coefs R{i+1}", value=",".join(["1"]*int(n_vars_e)), key=f"er_coef{i}")
            with rc:
                bi = st.number_input(f"RHS {i+1}", value=10.0, key=f"er_rhs{i}")
            rest_e.append(ri); rhs_e.append(bi)
        vars_enteras_sel = None
        if not todas_enteras:
            opts = [f"X{i+1}" for i in range(int(n_vars_e))]
            sel = st.multiselect("Variables enteras", opts, default=opts[:1], key="e_sel")
            vars_enteras_sel = [int(s[1:])-1 for s in sel]
        if st.button("Resolver PL Entera", key="btn_entera"):
            try:
                fo = _parse_fo(fo_e, int(n_vars_e))
                A = [_parse_fo(r, int(n_vars_e)) for r in rest_e]
                res = resolver_programacion_lineal_entera(fo, A, list(rhs_e), vars_enteras_sel if not todas_enteras else None, tipo_e)
                if "Óptimo" in res["Estado"]:
                    st.success(f"Z = {res['Valor_Optimo']}")
                    st.dataframe(pd.DataFrame(res["Variables"].items(), columns=["Variable", "Valor"]), use_container_width=True)
                else:
                    st.warning(res.get("Mensaje", ""))
            except Exception as e:
                st.error(f"Error: {e}")

    with tabs[3]:
        st.subheader("Análisis de Sensibilidad")
        tipo_as = st.selectbox("Tipo", ["max", "min"], key="as_tipo")
        n_vars_as = st.number_input("Variables", 2, 8, 2, key="as_nv")
        n_rest_as = st.number_input("Restricciones", 1, 10, 2, key="as_nr")
        fo_as = st.text_input("Función objetivo", value="5,4", key="as_fo")
        rest_as, rhs_as = [], []
        for i in range(int(n_rest_as)):
            cc, rc = st.columns([3, 1])
            with cc:
                ri = st.text_input(f"Coefs R{i+1}", value=",".join(["1"]*int(n_vars_as)), key=f"as_c{i}")
            with rc:
                bi = st.number_input(f"RHS {i+1}", value=6.0, key=f"as_r{i}")
            rest_as.append(ri); rhs_as.append(bi)
        if st.button("Analizar sensibilidad", key="btn_sens"):
            try:
                fo = _parse_fo(fo_as, int(n_vars_as))
                A = [_parse_fo(r, int(n_vars_as)) for r in rest_as]
                res = analizar_sensibilidad(fo, A, list(rhs_as), tipo_as)
                base = res["Solucion_Base"]
                if base["Estado"] == "Óptimo":
                    st.success(f"Z = {base['Valor_Optimo']}")
                    st.dataframe(pd.DataFrame(base["Variables"].items(), columns=["Variable", "Valor"]), use_container_width=True)
                    st.markdown("#### Sensibilidad FO")
                    st.dataframe(res["Sensibilidad_FO"], use_container_width=True)
                    st.markdown("#### Sensibilidad RHS")
                    df_rhs = res["Sensibilidad_RHS"].copy()
                    df_rhs["Holgura"] = res["Holguras"]
                    st.dataframe(df_rhs, use_container_width=True)
                else:
                    st.warning(base.get("Mensaje", ""))
            except Exception as e:
                st.error(f"Error: {e}")

    with tabs[4]:
        st.subheader("Problema de Transporte")
        col_m, col_n = st.columns(2)
        with col_m:
            m_t = int(st.number_input("Orígenes", 2, 6, 3, key="t_m"))
        with col_n:
            n_t = int(st.number_input("Destinos", 2, 6, 3, key="t_n"))
        oferta = [st.number_input(f"Oferta O{i+1}", value=30.0, key=f"to{i}") for i in range(m_t)]
        demanda = [st.number_input(f"Demanda D{j+1}", value=30.0, key=f"td{j}") for j in range(n_t)]
        costos = []
        for i in range(m_t):
            fila = []
            cols_c = st.columns(n_t)
            for j in range(n_t):
                with cols_c[j]:
                    fila.append(st.number_input(f"c[{i+1},{j+1}]", value=float((i+1)*(j+1)), key=f"tc{i}{j}"))
            costos.append(fila)
        if st.button("Resolver Transporte", key="btn_transp"):
            try:
                res = resolver_transporte([float(o) for o in oferta], [float(d) for d in demanda], costos)
                if res["Estado"] == "Óptimo":
                    st.success(f"Costo total = {res.get('Costo_Optimo', 'N/A')}")
                    # Mostrar asignación en un DataFrame si está disponible
                    asignacion = res.get("Asignacion")
                    if asignacion is not None:
                        st.dataframe(pd.DataFrame(asignacion), use_container_width=True)
                else:
                    st.warning(res.get("Mensaje", "No se encontró solución óptima."))
            except Exception as e:
                st.error(f"Error: {e}")
    with tabs[5]:
        st.subheader("Problema de Asignación")
        n_a = int(st.number_input("Dimensión n", 2, 8, 3, key="a_n"))
        tipo_a = st.selectbox("Tipo", ["min", "max"], key="a_tipo")
        st.markdown("**Matriz de costos/beneficios:**")
        costos_a = []
        for i in range(n_a):
            fila = []
            for j in range(n_a):
                val = st.number_input(
                    f"Agente {i+1} → Tarea {j+1}",
                    value=float(i*n_a+j+1),
                    key=f"ac{i}{j}"
                )
                fila.append(val)
            costos_a.append(fila)
        if st.button("Resolver Asignación", key="btn_asig"):
            try:
                res = resolver_asignacion(costos_a, tipo_a)
                if res["Estado"] == "Óptimo":
                    st.success(f"Valor óptimo = {res['Valor_Optimo']}")
                    st.dataframe(pd.DataFrame(res["Asignaciones"], columns=["Agente","Tarea","Costo/Beneficio"]), use_container_width=True)
                    st.dataframe(res["Tabla"], use_container_width=True)
                else:
                    st.warning(res["Mensaje"])
            except Exception as e:
                st.error(f"Error: {e}")