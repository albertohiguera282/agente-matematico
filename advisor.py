# -*- coding: utf-8 -*-
"""
Decision tree advisor logic for selecting statistical models.
"""

from database import MODEL_CATALOG

# Question structures
QUESTIONS = {
    "objective": {
        "title": "1. Objetivo Principal del Estudio",
        "desc": "¿Qué desea lograr principalmente con su análisis de datos?",
        "options": [
            ("describir", "Describir características de variables (Resumen / Localización)"),
            ("comparar", "Comparar diferencias entre grupos o condiciones"),
            ("explicar", "Explicar relaciones de causa/efecto o realizar predicciones"),
            ("asociar", "Evaluar fuerza de asociación o correlación (no causal)"),
            ("clasificar", "Clasificar observaciones o reducir dimensiones (Multivariado)"),
            ("timeseries", "Analizar tendencias y realizar pronósticos temporales"),
            ("survival", "Analizar el tiempo hasta que ocurra un evento (Supervivencia)")
        ]
    },
    "var_type": {
        "title": "2. Tipo de Variables Principales",
        "desc": "¿Cuál es la naturaleza de sus variables bajo estudio?",
        "options": [
            ("numeric", "Cuantitativas / Numéricas (Escala, Razón o Intervalo)"),
            ("categorical", "Cualitativas / Categóricas (Nominal u Ordinal)"),
            ("mixed", "Mixtas (Combinación de cuantitativas y cualitativas)")
        ]
    },
    "var_count": {
        "title": "3. Cantidad de Variables",
        "desc": "¿Cuántas variables están involucradas en este análisis específico?",
        "options": [
            ("1", "Una sola variable (Análisis Univariado)"),
            ("2", "Dos variables (Análisis Bivariado)"),
            ("3+", "Tres o más variables (Análisis Multivariado)")
        ]
    },
    "independence": {
        "title": "4. Relación / Independencia de las Muestras",
        "desc": "¿Cómo fueron recolectados los datos entre los grupos?",
        "options": [
            ("independent", "Muestras Independientes (Grupos conformados por sujetos distintos)"),
            ("paired", "Muestras Pareadas / Medidas Repetidas (Mismo sujeto medido en momentos distintos)")
        ]
    },
    "assumptions": {
        "title": "5. Cumplimiento de Supuestos Estadísticos",
        "desc": "¿Los datos cuantitativos cumplen con normalidad y homocedasticidad?",
        "options": [
            ("parametric", "Sí - Procedimiento Paramétrico (Normalidad y varianzas homogéneas)"),
            ("nonparametric", "No - Procedimiento No Paramétrico (Sin supuestos de distribución o escala ordinal)")
        ]
    }
}

# ------------------------------------------------------------
# Guía de aplicación por modelo
# ------------------------------------------------------------
# Para cada modelo: en qué momento del estudio se opera, un ejercicio específico
# y su aplicación. Las claves están en inglés (las usa app.py); el texto visible
# está en español. Los ejemplos usan ventas_200.csv (200 filas).
# Objetivo 1 "describir": media/desviación, mediana/IQR y frecuencias.
APPLICATION_GUIDE = {
    "arithmetic_mean": {
        "moment": (
            "Primer paso de la exploración, antes de cualquier prueba de hipótesis o modelo. "
            "Se opera sobre una variable numérica con distribución aproximadamente simétrica "
            "y sin valores extremos que la distorsionen."
        ),
        "exercise": (
            "Con `ventas_200.csv`: resumir `cantidad` (media ≈ 2.10, mediana 2.00, desv. est. ≈ 1.36, N = 200). "
            "Como la media y la mediana casi coinciden, la media representa bien cuántas unidades se venden por orden."
        ),
        "application": (
            "Línea base de un estudio, control de calidad de un proceso, reporte de promedios con su "
            "intervalo de confianza al 95% y verificación de supuestos antes de una t de Student, "
            "un ANOVA o una regresión."
        ),
        "limits": (
            "Es sensible a valores extremos: si la media se aleja de la mediana, "
            "cambia a mediana e IQR."
        ),
    },
    "median_iqr_var1": {
        "moment": (
            "Mismo momento exploratorio, pero cuando la variable es asimétrica, tiene valores atípicos "
            "o es ordinal. Se opera después de mirar el histograma o el boxplot y de comparar la media "
            "con la mediana."
        ),
        "exercise": (
            "Con `ventas_200.csv`: resumir `total`. La media (≈ 141.6) queda muy por encima de la mediana (95.0) "
            "y el IQR es 175.25 (Q1 = 74.75, Q3 = 250.00): hay asimetría a la derecha, "
            "así que la mediana describe mejor la venta típica."
        ),
        "application": (
            "Ingresos, precios, tiempos de espera y escalas ordinales (tipo Likert), o cualquier reporte "
            "donde pocos valores altos distorsionarían el promedio."
        ),
        "limits": (
            "Usa menos información que la media. Si la distribución es simétrica, la media y la desviación "
            "estándar son más eficientes y permiten inferencia paramétrica."
        ),
    },
    "frequencies_props_var1": {
        "moment": (
            "Primer paso de la exploración de variables categóricas (nominales u ordinales), antes de "
            "cruzarlas con otras variables (Chi-cuadrado) o de usarlas como factor en una comparación."
        ),
        "exercise": (
            "Con `ventas_200.csv`: tabla de frecuencias de `region`. Norte 79 ventas (39.5%), Sur 41 (20.5%), "
            "Este 40 (20.0%) y Oeste 40 (20.0%): casi 4 de cada 10 ventas se concentran en el Norte."
        ),
        "application": (
            "Perfil de clientes, participación por región, canal o método de pago, y detección de categorías "
            "con muy pocos casos antes de un Chi-cuadrado (frecuencias esperadas bajas)."
        ),
        "limits": (
            "Solo resume conteos; no mide asociación. Para saber si dos variables categóricas se "
            "relacionan usa la prueba de Chi-cuadrado."
        ),
    },
    "oneway_anova": {
        "moment": (
            "Fase de contraste de hipótesis, cuando ya describiste los datos y quieres saber si la media de "
            "una variable numérica difiere entre 3 o más grupos independientes (categorías de un factor). "
            "Se opera después de revisar los supuestos: normalidad de los residuos y varianzas homogéneas."
        ),
        "exercise": (
            "Con `ventas_200.csv`: comparar `total` entre las categorías de `region` (Norte, Sur, Este, Oeste). "
            "Como `total` es asimétrica a la derecha (media ≈ 141.6 frente a mediana 95.0), mira primero "
            "la prueba de Shapiro-Wilk sobre los residuos y la de Levene que muestra la calculadora; si fallan, "
            "apóyate en Welch o en Kruskal-Wallis, que también se reportan. Si el F global es significativo, "
            "usa Tukey para saber qué pares de regiones difieren."
        ),
        "application": (
            "Comparar el rendimiento de varios tratamientos, métodos de enseñanza, sucursales, regiones o "
            "proveedores. Se reporta F, grados de libertad, p-valor y tamaño del efecto (η², ω²)."
        ),
        "limits": (
            "Solo dice que existe al menos una diferencia; para saber cuáles grupos difieren necesita un post-hoc. "
            "Exige independencia, residuos aproximadamente normales y varianzas homogéneas. "
            "Con solo 2 grupos usa la prueba t; con medidas repetidas usa ANOVA de medidas repetidas."
        ),
    },
}


def get_application_guide(model_id):
    """Devuelve la guía (momento, ejercicio, aplicación y límites) de un modelo, o None si no existe."""
    return APPLICATION_GUIDE.get(model_id)


def get_recommendation(answers):
    """
    Evaluates answers and returns the recommended model, scientific justification,
    assumptions, and links to alternative models.
    """
    obj = answers.get("objective")
    v_type = answers.get("var_type")
    v_count = answers.get("var_count")
    ind = answers.get("independence", "independent")
    param = answers.get("assumptions", "parametric")

    rec_id = "arithmetic_mean"  # default
    alt_id = None

    # 1. Descriptive
    if obj == "describir":
        if v_type == "categorical":
            rec_id = "frequencies_props_var1"
        else:
            if param == "parametric":
                rec_id = "arithmetic_mean"
                alt_id = "median_iqr_var1"
            else:
                rec_id = "median_iqr_var1"
                alt_id = "arithmetic_mean"

    # 2. Compare
    elif obj == "comparar":
        if v_type == "numeric":
            if v_count == "2":
                if ind == "independent":
                    if param == "parametric":
                        rec_id = "independent_ttest"
                        alt_id = "wilcoxon_mann_whitney_var1"
                    else:
                        rec_id = "wilcoxon_mann_whitney_var1"
                        alt_id = "independent_ttest"
                else: # paired
                    if param == "parametric":
                        rec_id = "paired_ttest"
                        alt_id = "wilcoxon_mann_whitney_var3"  # Wilcoxon Signed Rank (robust/paired bootstrap variant)
                    else:
                        rec_id = "wilcoxon_mann_whitney_var3"
                        alt_id = "paired_ttest"
            else: # 3+ variables (ANOVA-like)
                if ind == "independent":
                    if param == "parametric":
                        rec_id = "oneway_anova"
                        alt_id = "kmeans_var3" # robust non-parametric groupings
                    else:
                        rec_id = "wilcoxon_mann_whitney_var2" # Kruskal-Wallis equivalent
                        alt_id = "oneway_anova"
                else: # Repeated measures
                    rec_id = "repeated_measures_anova_var1"
                    alt_id = "repeated_measures_anova_var3"
        elif v_type == "categorical":
            rec_id = "chi_squared_test"
            alt_id = "chi_squared_test_var3"  # Bootstrap Chi-squared
        else: # Mixed
            rec_id = "ancova_var1"
            alt_id = "ancova_var3"

    # 3. Explain/Predict
    elif obj == "explicar":
        if v_type == "numeric":
            if v_count == "2":
                rec_id = "simple_regression"
                alt_id = "simple_regression_var3" # Bootstrap simple regression
            else:
                rec_id = "multiple_regression_var1"
                alt_id = "multiple_regression_var7" # Robust Huber regression
        elif v_type == "categorical":
            rec_id = "binary_logistic_var1"
            alt_id = "decision_tree_var1"
        else: # Mixed
            if v_count == "2":
                rec_id = "simple_regression"
                alt_id = "binary_logistic_var1"
            else:
                rec_id = "multiple_regression_var1"
                alt_id = "random_forest_var1"

    # 4. Association / Correlation
    elif obj == "asociar":
        if v_type == "categorical":
            rec_id = "chi_squared_test"
            alt_id = "chi_squared_test_var3"
        else: # Numeric
            if param == "parametric":
                rec_id = "pearson_corr_var1"
                alt_id = "spearman_corr_var1"
            else:
                rec_id = "spearman_corr_var1"
                alt_id = "pearson_corr_var1"

    # 5. Classify / Reduce
    elif obj == "clasificar":
        if v_type == "numeric":
            rec_id = "pca_var1"
            alt_id = "kmeans_var1"
        else:
            rec_id = "hierarchical_cluster_var1"
            alt_id = "kmeans_var1"

    # 6. Time series
    elif obj == "timeseries":
        rec_id = "arima_var1"
        alt_id = "holt_winters_var1"

    # 7. Survival
    elif obj == "survival":
        rec_id = "kaplan_meier_var1"
        alt_id = "cox_regression_var1"

    # Search in catalog
    model_info = next((m for m in MODEL_CATALOG if m["id"] == rec_id), None)
    alt_info = next((m for m in MODEL_CATALOG if m["id"] == alt_id), None) if alt_id else None

    # If not found (fallback to base)
    if not model_info:
        model_info = MODEL_CATALOG[0]

    return {
        "model": model_info,
        "alternative": alt_info,
        "justification": model_info.get("justification", ""),
        "assumptions": model_info.get("assumptions", []),
        "guide": get_application_guide(model_info.get("id")),
        "alt_guide": get_application_guide(alt_info.get("id")) if alt_info else None,
    }
