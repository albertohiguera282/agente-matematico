import sympy as sp
import streamlit as st

def renderizar_modulo_polinomios():
    st.subheader("🧬 Análisis de Polinomios, Factorización y Raíces")
    
    # Cajón de texto interactivo para el usuario
    polinomio_input = st.text_input(
        "Ingresa un polinomio en x (ej: x**2 + (n-1)*x + 9 o x**3 - 6*x**2 + 11*x - 6):",
        value="x**2 + (n-1)*x + 9",
        key="input_polinomio_user"
    )
    
    if st.button("Analizar Polinomio y Hallar Raíces", key="btn_analizar_poli"):
        # Limpieza previa de la cadena para evitar errores de sintaxis comunes
        entrada_limpia = polinomio_input.strip()
        
        # Si el usuario escribe f(x) = ... o y = ..., nos quedamos solo con la expresión derecha
        if "=" in entrada_limpia:
            entrada_limpia = entrada_limpia.split("=")[-1].strip()
            
        # Reemplazos amigables por si se ingresan expresiones tipo texto clásico
        entrada_limpia = entrada_limpia.replace("^", "**")  # Convierte x^2 a x**2
        
        try:
            # Declaración explícita de los símbolos analíticos multivariables (x, n)
            x, n = sp.symbols('x n')
            
            # Conversión de la cadena de texto a expresión matemática abstracta de SymPy
            expresion = sp.sympify(entrada_limpia)
            
            st.success("🎯 ¡Polinomio parseado analíticamente con éxito!")
            
            # --- 1. MOSTRAR EXPRESIÓN FORMAL ---
            st.markdown("### 📝 Expresión Matemática Simplificada:")
            st.latex(sp.latex(expresion))
            
            # --- 2. FACTORIZACIÓN ---
            st.markdown("### 📂 Factorización en el Campo Complejo/Real:")
            try:
                forma_factorizada = sp.factor(expresion)
                st.latex(f"f(x) = {sp.latex(forma_factorizada)}")
            except Exception:
                st.info("La expresión no posee una factorización exacta con coeficientes enteros.")
                
            # --- 3. CÁLCULO DE RAÍCES ANALÍTICAS ---
            st.markdown("### 🔑 Raíces o Soluciones del Polinomio ($f(x) = 0$):")
            # Resuelve la ecuación respecto a x, manteniendo n como parámetro simbólico
            raices = sp.solve(expresion, x)
            
            if raices:
                for i, raiz in enumerate(raices, start=1):
                    st.markdown(f"**Raíz $x_{i}$:**")
                    st.latex(f"x_{i} = {sp.latex(raiz)}")
            else:
                st.warning("No se encontraron soluciones analíticas explícitas para esta configuración.")
                
        except Exception as e:
            st.error(f"❌ Error procesando el polinomio: {e}")
            st.info("Asegúrate de colocar los asteriscos (*) explícitamente para las multiplicaciones (ej: 2*x en lugar de 2x).")


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
