import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def analizar_funcion(expr_str, var_str="x"):
    """
    Analiza una función f(x): calcula derivada, integral indefinida y puntos críticos.
    """
    try:
        x = sp.Symbol(var_str)
        f = sp.sympify(expr_str)
        
        derivada_1 = sp.diff(f, x)
        derivada_2 = sp.diff(derivada_1, x)
        
        integral = sp.integrate(f, x)
        
        # Intentar hallar puntos críticos (f'(x) = 0)
        puntos_criticos = []
        try:
            puntos_criticos = sp.solve(derivada_1, x)
            # Filtrar solo valores reales
            puntos_criticos = [sp.N(p) for p in puntos_criticos if p.is_real]
        except:
            puntos_criticos = "Complejo/No analítico"
            
        return {
            "Original": f,
            "Derivada_1": derivada_1,
            "Derivada_2": derivada_2,
            "Integral_Indefinida": integral,
            "Puntos_Criticos": puntos_criticos
        }
    except Exception as e:
        raise ValueError(f"Error analizando la función: {e}")

def graficar_funcion_y_derivada(expr_str, var_str="x", x_lims=(-10, 10)):
    """
    Genera un gráfico de f(x) y su derivada f'(x).
    """
    try:
        x_sym = sp.Symbol(var_str)
        f_sym = sp.sympify(expr_str)
        f_prime_sym = sp.diff(f_sym, x_sym)
        
        # Convertir a funciones evaluables numéricamente
        f_num = sp.lambdify(x_sym, f_sym, "numpy")
        f_prime_num = sp.lambdify(x_sym, f_prime_sym, "numpy")
        
        # Mapear valores
        x_vals = np.linspace(x_lims[0], x_lims[1], 400)
        
        # Evaluar (con control de divisiones por cero en lambdify)
        try:
            y_vals = f_num(x_vals)
            # Si f es constante, lambdify puede retornar un escalar en vez de un array
            if np.isscalar(y_vals):
                y_vals = np.full_like(x_vals, y_vals)
        except Exception:
            y_vals = [float(f_sym.subs(x_sym, val)) for val in x_vals]
            
        try:
            y_prime_vals = f_prime_num(x_vals)
            if np.isscalar(y_prime_vals):
                y_prime_vals = np.full_like(x_vals, y_prime_vals)
        except Exception:
            y_prime_vals = [float(f_prime_sym.subs(x_sym, val)) for val in x_vals]
            
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_vals, y_vals, label=f"$f(x) = {sp.latex(f_sym)}$", color="#9B5DE5", linewidth=2.5)
        ax.plot(x_vals, y_prime_vals, label=f"$f'(x) = {sp.latex(f_prime_sym)}$", color="#00F5D4", linewidth=2, linestyle="--")
        
        ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.grid(color="#333333", linestyle=":", alpha=0.6)
        
        ax.set_title("Graficador de Función y Derivada", color="white", fontsize=12)
        ax.legend(facecolor="#1e1e24", edgecolor="#9B5DE5", labelcolor="white")
        
        # Estilos oscuros para matploblit
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1E1E24")
        ax.tick_params(colors="white")
        ax.spines['bottom'].set_color('#333333')
        ax.spines['top'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.spines['right'].set_color('#333333')
        
        return fig
    except Exception as e:
        raise ValueError(f"Error graficando función: {e}")

def calcular_integral_definida(expr_str, a, b, var_str="x"):
    """
    Calcula la integral definida de f(x) de a a b y genera el gráfico con el área sombreada.
    """
    try:
        x_sym = sp.Symbol(var_str)
        f_sym = sp.sympify(expr_str)
        
        # Cálculo analítico y numérico
        integral_analitica = sp.integrate(f_sym, (x_sym, a, b))
        valor_numerico = float(sp.N(integral_analitica))
        
        # Generar gráfico con área sombreada
        f_num = sp.lambdify(x_sym, f_sym, "numpy")
        
        # Rango amplio para el gráfico
        ancho = abs(b - a)
        x_min = a - max(ancho * 0.5, 2.0)
        x_max = b + max(ancho * 0.5, 2.0)
        
        x_vals = np.linspace(x_min, x_max, 400)
        
        try:
            y_vals = f_num(x_vals)
            if np.isscalar(y_vals):
                y_vals = np.full_like(x_vals, y_vals)
        except Exception:
            y_vals = [float(f_sym.subs(x_sym, val)) for val in x_vals]
            
        # Valores para el área sombreada
        x_area = np.linspace(a, b, 200)
        try:
            y_area = f_num(x_area)
            if np.isscalar(y_area):
                y_area = np.full_like(x_area, y_area)
        except Exception:
            y_area = [float(f_sym.subs(x_sym, val)) for val in x_area]
            
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_vals, y_vals, label=f"$f(x) = {sp.latex(f_sym)}$", color="#9B5DE5", linewidth=2.5)
        ax.fill_between(x_area, y_area, 0, where=None, interpolate=True, color="#00F5D4", alpha=0.3, label="Área de integración")
        
        ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.grid(color="#333333", linestyle=":", alpha=0.6)
        
        ax.set_title(f"Integral Definida de {a} a {b}", color="white", fontsize=12)
        ax.legend(facecolor="#1e1e24", edgecolor="#00F5D4", labelcolor="white")
        
        # Estilos oscuros para matploblit
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1E1E24")
        ax.tick_params(colors="white")
        ax.spines['bottom'].set_color('#333333')
        ax.spines['top'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.spines['right'].set_color('#333333')
        
        return {
            "Valor_Exacto": integral_analitica,
            "Valor_Numerico": valor_numerico,
            "Fig": fig
        }
    except Exception as e:
        raise ValueError(f"Error calculando la integral definida: {e}")
