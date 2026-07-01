import itertools
import re

def evaluar_expresion(formula_original, contexto):
    expr = formula_original
    for var, val in sorted(contexto.items(), key=lambda x: len(x[0]), reverse=True):
        expr = re.sub(r'\b' + re.escape(var) + r'\b', str(val), expr)
    expr = expr.replace('<->', '==')
    # A -> B = (not A or B)
    for _ in range(15):
        nuevo = re.sub(
            r'\(([^()]+?)\)\s*->\s*\(([^()]+?)\)',
            lambda m: '(not (' + m.group(1) + ') or (' + m.group(2) + '))',
            expr
        )
        nuevo = re.sub(
            r'(True|False)\s*->\s*(True|False)',
            lambda m: '(not ' + m.group(1) + ' or ' + m.group(2) + ')',
            nuevo
        )
        if nuevo == expr:
            break
        expr = nuevo
    expr = expr.replace('^', ' and ')
    expr = re.sub(r'\bv\b', ' or ', expr)
    expr = re.sub(r'[~\u00ac]', ' not ', expr)
    try:
        return bool(eval(expr, {"__builtins__": None}, {"True": True, "False": False}))
    except Exception as e:
        raise ValueError(f"Error evaluando '{expr}': {e}")

def generar_tabla_verdad(formula_original):
    clean = re.sub(r'->|<->', 'X', formula_original)
    variables = sorted(set(re.findall(r'\b[A-Z]\b', clean)))
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
