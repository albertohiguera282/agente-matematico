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
            ("describir", "Describir caracteristicas de variables (Resumen / Localización)"),
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
        "assumptions": model_info.get("assumptions", [])
    }
