import sympy as sp
def resolver_matriz_operaciones(matriz_list):
    """
    Realiza operaciones básicas sobre una matriz usando SymPy (cálculos exactos).
    matriz_list: lista bidimensional de números o expresiones
    """
    try:
        M = sp.Matrix(matriz_list)
        resultados = {
            "Original": M,
            "Transpuesta": M.T,
            "Dimensiones": M.shape,
        }
        
        # Verificar si es cuadrada para calcular determinante e inversa
        if M.shape[0] == M.shape[1]:
            det = M.det()
            resultados["Determinante"] = det
            if det != 0:
                resultados["Inversa"] = M.inv()
            else:
                resultados["Inversa"] = "No invertible (Determinante = 0)"
        else:
            resultados["Determinante"] = "No aplica (Debe ser cuadrada)"
            resultados["Inversa"] = "No aplica (Debe ser cuadrada)"
            
        # Forma escalonada reducida por filas (RREF) y pivotes
        rref_matrix, pivots = M.rref()
        resultados["RREF"] = rref_matrix
        resultados["Pivotes"] = pivots
        
        return resultados
    except Exception as e:
        raise ValueError(f"Error procesando la matriz: {e}")
def resolver_sistema_lineal(ecuaciones_list, variables_str="x, y, z"):
    """
    Resuelve un sistema de ecuaciones lineales representadas por cadenas de texto.
    ecuaciones_list: lista de strings de ecuaciones, ej: ["2*x + y - z - 8", "x - y + z - 1", "3*x + 2*y + z - 13"]
    (las ecuaciones se asumen igualadas a cero o pueden contener "=")
    """
    try:
        vars_list = sp.symbols(variables_str)
        eqs = []
        for eq_str in ecuaciones_list:
            if not eq_str.strip():
                continue
            if "=" in eq_str:
                lhs, rhs = eq_str.split("=")
                eqs.append(sp.sympify(lhs.strip()) - sp.sympify(rhs.strip()))
            else:
                eqs.append(sp.sympify(eq_str.strip()))
        
        soluciones = sp.solve(eqs, vars_list)
        return {
            "Soluciones": soluciones,
            "Variables": vars_list,
            "Ecuaciones_Sympy": eqs
        }
    except Exception as e:
        raise ValueError(f"Error resolviendo el sistema: {e}")
def resolver_polinomio(expresion_str, var_str="x"):
    """
    Calcula factorización, raíces y expansión de un polinomio.
    """
    try:
        x = sp.Symbol(var_str)
        expr = sp.sympify(expresion_str)
        
        factores = sp.factor(expr)
        raices = sp.solve(expr, x)
        derivada = sp.diff(expr, x)
        
        return {
            "Original": expr,
            "Factorizacion": factores,
            "Raices": raices,
            "Derivada": derivada
        }
    except Exception as e:
        raise ValueError(f"Error procesando el polinomio: {e}")
