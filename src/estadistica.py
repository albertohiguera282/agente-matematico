# Contenido definitivo y único de estadistica.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def descriptiva(datos):
    """Calcula estadísticas descriptivas (centralización, dispersión, cuartiles) de una lista de datos."""
    datos = np.array(datos, dtype=float)
    n = len(datos)

    media = float(np.mean(datos))
    mediana = float(np.median(datos))

    # Moda (puede haber más de una o ninguna si todos los valores son distintos)
    valores, conteos = np.unique(datos, return_counts=True)
    max_conteo = conteos.max()
    if max_conteo == 1:
        moda = "No hay moda (todos los valores son únicos)"
    else:
        modas = valores[conteos == max_conteo]
        moda = ", ".join(str(round(m, 4)) for m in modas)

    minimo = float(np.min(datos))
    maximo = float(np.max(datos))
    rango = maximo - minimo

    if n > 1:
        varianza = float(np.var(datos, ddof=1))
        desviacion_estandar = float(np.std(datos, ddof=1))
        coef_variacion = round((desviacion_estandar / media) * 100, 2) if media != 0 else "Indefinido (media = 0)"
    else:
        varianza = "No definida (se requiere más de un dato)"
        desviacion_estandar = "No definida"
        coef_variacion = "No definido"

    q1 = float(np.percentile(datos, 25))
    q2 = float(np.percentile(datos, 50))
    q3 = float(np.percentile(datos, 75))
    p10 = float(np.percentile(datos, 10))
    p90 = float(np.percentile(datos, 90))

    return {
        "Cantidad": n,
        "Media": media,
        "Mediana": round(mediana, 4),
        "Moda": moda,
        "Minimo": round(minimo, 4),
        "Maximo": round(maximo, 4),
        "Rango": round(rango, 4),
        "Varianza": varianza,
        "Desviacion_Estandar": desviacion_estandar,
        "Coeficiente_Variacion (%)": coef_variacion,
        "Q1": round(q1, 4),
        "Q2": round(q2, 4),
        "Q3": round(q3, 4),
        "P10": p10,
        "P90": p90
    }


def analizar_distribucion_normal(mu, sigma, x_val, operacion):
    """Calcula probabilidad acumulada para una distribución normal y genera el gráfico."""
    if operacion == "<=":
        probabilidad = stats.norm.cdf(x_val, loc=mu, scale=sigma)
        titulo_formula = f"P(X \\le {x_val})"
    else:  # ">="
        probabilidad = 1 - stats.norm.cdf(x_val, loc=mu, scale=sigma)
        titulo_formula = f"P(X \\ge {x_val})"

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    y = stats.norm.pdf(x, loc=mu, scale=sigma)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, color="#00F5D4", linewidth=2)

    if operacion == "<=":
        x_fill = x[x <= x_val]
    else:
        x_fill = x[x >= x_val]
    y_fill = stats.norm.pdf(x_fill, loc=mu, scale=sigma)
    ax.fill_between(x_fill, y_fill, color="#9B5DE5", alpha=0.6)

    ax.set_title("Distribución Normal", color="white", fontsize=10)
    ax.set_facecolor("#1E1E24")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.grid(True, linestyle=":", alpha=0.4, color="#333333")

    return {
        "Probabilidad": float(probabilidad),
        "Titulo_Formula": titulo_formula,
        "Fig": fig
    }


def analizar_distribucion_binomial(n, p, k, operacion):
    """Calcula probabilidad para una distribución binomial y genera el gráfico."""
    if operacion == "=":
        probabilidad = stats.binom.pmf(k, n, p)
        titulo_formula = f"P(X = {k})"
    elif operacion == "<=":
        probabilidad = stats.binom.cdf(k, n, p)
        titulo_formula = f"P(X \\le {k})"
    else:  # ">="
        probabilidad = 1 - stats.binom.cdf(k - 1, n, p)
        titulo_formula = f"P(X \\ge {k})"

    valores_x = np.arange(0, n + 1)
    valores_y = stats.binom.pmf(valores_x, n, p)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colores = []
    for xi in valores_x:
        if operacion == "=":
            resaltar = (xi == k)
        elif operacion == "<=":
            resaltar = (xi <= k)
        else:
            resaltar = (xi >= k)
        colores.append("#9B5DE5" if resaltar else "#00F5D4")

    ax.bar(valores_x, valores_y, color=colores, edgecolor="black", alpha=0.85)
    ax.set_title("Distribución Binomial", color="white", fontsize=10)
    ax.set_xlabel("k (número de éxitos)", color="white")
    ax.set_ylabel("P(X = k)", color="white")
    ax.set_facecolor("#1E1E24")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.grid(True, linestyle=":", alpha=0.4, color="#333333")

    return {
        "Probabilidad": float(probabilidad),
        "Titulo_Formula": titulo_formula,
        "Fig": fig
    }


def calcular_intervalo_confianza(datos, confianza):
    """Calcula el intervalo de confianza para la media usando la distribución t-Student."""
    datos = np.array(datos, dtype=float)
    n = len(datos)

    if n < 2:
        return "Se requieren al menos 2 datos para calcular un intervalo de confianza."

    media = float(np.mean(datos))
    error_estandar = stats.sem(datos)
    grados_libertad = n - 1

    margen_error = stats.t.ppf((1 + confianza) / 2, grados_libertad) * error_estandar

    limite_inferior = media - margen_error
    limite_superior = media + margen_error

    return {
        "Confianza_Pct": round(confianza * 100, 2),
        "Media": media,
        "Margen_Error": float(margen_error),
        "Limite_Inferior": float(limite_inferior),
        "Limite_Superior": float(limite_superior)
    }


def realizar_prueba_hipotesis_t(datos, mu_h0, cola, alpha):
    """Realiza una prueba t de hipótesis para una muestra."""
    datos = np.array(datos, dtype=float)
    n = len(datos)

    if n < 2:
        return "Se requieren al menos 2 datos para realizar la prueba de hipótesis."

    media_muestral = float(np.mean(datos))
    desviacion = np.std(datos, ddof=1)
    grados_libertad = n - 1
    error_estandar = desviacion / np.sqrt(n)

    estadistico_t = (media_muestral - mu_h0) / error_estandar

    if cola == "dos-colas":
        p_valor = 2 * (1 - stats.t.cdf(abs(estadistico_t), grados_libertad))
        h0_texto = f"\\mu = {mu_h0}"
        h1_texto = f"\\mu \\neq {mu_h0}"
    elif cola == "menor":
        p_valor = stats.t.cdf(estadistico_t, grados_libertad)
        h0_texto = f"\\mu \\ge {mu_h0}"
        h1_texto = f"\\mu < {mu_h0}"
    else:  # "mayor"
        p_valor = 1 - stats.t.cdf(estadistico_t, grados_libertad)
        h0_texto = f"\\mu \\le {mu_h0}"
        h1_texto = f"\\mu > {mu_h0}"

    decision = "Rechazar H0" if p_valor < alpha else "No rechazar H0"

    return {
        "H0": h0_texto,
        "H1": h1_texto,
        "Media_Muestral": media_muestral,
        "Grados_Libertad": grados_libertad,
        "Estadistico_t": float(estadistico_t),
        "p_valor": float(p_valor),
        "Alpha": alpha,
        "Decision": decision
    }


def calcular_tamano_muestra(tipo, poblacion, error_margen, sigma_o_proporcion, confianza):
    """Calcula el tamaño de muestra requerido para estimar una media o una proporción."""
    z = stats.norm.ppf((1 + confianza) / 2)

    if tipo == "media":
        n0 = (z * sigma_o_proporcion / error_margen) ** 2
    else:  # "proporción"
        p = sigma_o_proporcion
        n0 = (z ** 2 * p * (1 - p)) / (error_margen ** 2)

    if poblacion:
        n_ajustado = n0 / (1 + (n0 - 1) / poblacion)
        return int(np.ceil(n_ajustado))

    return int(np.ceil(n0))
