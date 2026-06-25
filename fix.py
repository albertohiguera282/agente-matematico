# fix.py — inserta bloque de Lógica en app.py y reescribe logica.py

# ── 1. Reescribir logica.py ─────────────────────────────────────
logica_code = """import itertools
import re

def evaluar_expresion(formula_original, contexto):
    expr = formula_original
    for var, val in sorted(contexto.items(), key=lambda x: len(x[0]), reverse=True):
        expr = re.sub(r'\\b' + re.escape(var) + r'\\b', str(val), expr)
    expr = expr.replace('<->', '==')
    # A -> B = (not A or B)
    for _ in range(15):
        nuevo = re.sub(
            r'\\(([^()]+?)\\)\\s*->\\s*\\(([^()]+?)\\)',
            lambda m: '(not (' + m.group(1) + ') or (' + m.group(2) + '))',
            expr
        )
        nuevo = re.sub(
            r'(True|False)\\s*->\\s*(True|False)',
            lambda m: '(not ' + m.group(1) + ' or ' + m.group(2) + ')',
            nuevo
        )
        if nuevo == expr:
            break
        expr = nuevo
    expr = expr.replace('^', ' and ')
    expr = re.sub(r'\\bv\\b', ' or ', expr)
    expr = re.sub(r'[~\\u00ac]', ' not ', expr)
    try:
        return bool(eval(expr, {"__builtins__": None}, {"True": True, "False": False}))
    except Exception as e:
        raise ValueError(f"Error evaluando '{expr}': {e}")

def generar_tabla_verdad(formula_original):
    clean = re.sub(r'->|<->', 'X', formula_original)
    variables = sorted(set(re.findall(r'\\b[A-Z]\\b', clean)))
    palabras = {"AND", "OR", "NOT", "TRUE", "FALSE"}
    variables = [v for v in variables if v not in palabras]
    if not variables:
        raise ValueError("No se detectaron variables (usa letras mayusculas: P, Q, R...).")
    combinaciones = list(itertools.product([True, False], repeat=len(variables)))
    tabla, resultados = [], []
    for comb in combinaciones:
        contexto = dict(zip(variables, comb))
        resultado = evaluar_expresion(formula_original, contexto)
        fila = {var: ("V" if val else "F") for var, val in contexto.items()}
        fila["Resultado"] = "V" if resultado else "F"
        tabla.append(fila)
        resultados.append(resultado)
    if all(resultados):
        clasificacion = "Tautologia (Siempre Verdadera)"
    elif not any(resultados):
        clasificacion = "Contradiccion (Siempre Falsa)"
    else:
        clasificacion = "Contingencia (Depende de las variables)"
    return {"Variables": variables, "Tabla": tabla, "Clasificacion": clasificacion}
"""

with open("logica.py", "w", encoding="utf-8") as f:
    f.write(logica_code)
print("logica.py reescrito")

# ── 2. Insertar bloque de Logica en app.py ──────────────────────
bloque = """# ----------------- 5. LOGICA MATEMATICA -----------------
elif rama_seleccionada == "\U0001f9e0 L\u00f3gica Matem\u00e1tica":
    st.subheader("Generador de Tablas de Verdad Proposicionales")
    st.write("Ingresa una proposici\u00f3n l\u00f3gica usando letras may\u00fasculas (P, Q, R...).")
    st.write("Operadores: `^` AND | `v` OR | `~` NOT | `->` Condicional | `<->` Bicondicional")

    leyes = {
        "1. Identidad":          "P <-> P",
        "2. Doble negaci\u00f3n":     "P <-> ~(~P)",
        "3. Tercero excluido":   "P v ~P",
        "4. No contradicci\u00f3n":  "~(P ^ ~P)",
        "5. Conmutativa AND":    "(P ^ Q) <-> (Q ^ P)",
        "6. Conmutativa OR":     "(P v Q) <-> (Q v P)",
        "7. Asociativa AND":     "((P ^ Q) ^ R) <-> (P ^ (Q ^ R))",
        "8. Asociativa OR":      "((P v Q) v R) <-> (P v (Q v R))",
        "9. Distributiva 1":     "(P ^ (Q v R)) <-> ((P ^ Q) v (P ^ R))",
        "10. Distributiva 2":    "(P v (Q ^ R)) <-> ((P v Q) ^ (P v R))",
        "11. De Morgan 1":       "~(P ^ Q) <-> (~P v ~Q)",
        "12. De Morgan 2":       "~(P v Q) <-> (~P ^ ~Q)",
        "13. Absorci\u00f3n 1":       "(P v (P ^ Q)) <-> P",
        "14. Absorci\u00f3n 2":       "(P ^ (P v Q)) <-> P",
        "15. Implicaci\u00f3n":       "(P -> Q) <-> (~P v Q)",
        "16. Contraposici\u00f3n":    "(P -> Q) <-> (~Q -> ~P)",
        "17. Bicondicional":     "(P <-> Q) <-> ((P -> Q) ^ (Q -> P))",
    }

    st.markdown("#### Leyes L\u00f3gicas — haz clic para cargar")
    cols_l = st.columns(3)
    for idx, (nombre, formula) in enumerate(leyes.items()):
        with cols_l[idx % 3]:
            if st.button(nombre, key=f"ley_{idx}"):
                st.session_state["formula_logica_val"] = formula

    formula_logica = st.text_input(
        "Proposici\u00f3n l\u00f3gica:",
        value=st.session_state.get("formula_logica_val", "(P -> Q) ^ (~Q -> ~P)"),
        key="formula_logica_input"
    )

    if st.button("Generar Tabla de Verdad", key="btn_tabla_verdad"):
        try:
            res = logica.generar_tabla_verdad(formula_logica)
            st.success(f"Clasificaci\u00f3n: **{res['Clasificacion']}**")
            df_tabla = pd.DataFrame(res["Tabla"])
            cols_ord = res["Variables"] + ["Resultado"]
            st.table(df_tabla[cols_ord])
        except Exception as e:
            st.error(f"Error generando tabla de verdad: {e}")

"""

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Encontrar linea de FINANCIERA para insertar antes
insert_line = None
for i, line in enumerate(lines):
    if "FINANCIERA" in line and "elif" in line:
        insert_line = i
        break

if insert_line is None:
    # Buscar por el patron de finanzas
    for i, line in enumerate(lines):
        if "Matem\u00e1tica Financiera" in line and "elif" in line:
            insert_line = i
            break

if insert_line:
    patched = lines[:insert_line] + [bloque + "\n"] + lines[insert_line:]
    with open("app.py", "w", encoding="utf-8") as f:
        f.writelines(patched)
    print(f"app.py actualizado: bloque de Logica insertado antes de la linea {insert_line}")
else:
    print("ERROR: No se encontro la linea de Financiera para insertar el bloque")