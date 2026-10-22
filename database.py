
# -*- coding: utf-8 -*-
"""
Database of statistical models, assumptions, justifications, and code templates.
Generates 300+ models programmatically for academic research.
"""

import pandas as pd
from lote1_models import LOTE_1_MODELS

# Modules definition
MODULES = [
    {"id": "descriptive", "name": "1. Estadística Descriptiva", "desc": "Caracterización y resumen de conjuntos de datos."},
    {"id": "probability", "name": "2. Probabilidad y Distribuciones", "desc": "Modelado probabilístico y formas teóricas de distribución."},
    {"id": "inference", "name": "3. Inferencia Estadística", "desc": "Pruebas de hipótesis científicas y estimación de intervalos."},
    {"id": "regression", "name": "4. Modelos de Regresión", "desc": "Modelado de relaciones de dependencia e inferencia predictiva."},
    {"id": "correlation", "name": "5. Correlación y Asociación", "desc": "Medidas de dependencia no causal y tablas de contingencia."},
    {"id": "multivariate", "name": "6. Análisis Multivariado", "desc": "Métodos de reducción dimensional, agrupamiento y clasificación."},
    {"id": "timeseries", "name": "7. Series de Tiempo y Pronósticos", "desc": "Modelado de dependencia temporal y proyección analítica."},
    {"id": "anova", "name": "8. Diseño Experimental (ANOVA)", "desc": "Comparación múltiple de medias bajo diseño experimental estructurado."},
    {"id": "survival", "name": "9. Análisis de Supervivencia", "desc": "Análisis de tiempo hasta un evento de interés y tasas de riesgo."},
    {"id": "sem", "name": "10. Modelos de Ecuaciones Estructurales (SEM)", "desc": "Sistemas de ecuaciones simultáneas con variables latentes y observadas."},
    {"id": "advanced_econometrics", "name": "11. Modelos Avanzados y Econometría", "desc": "Técnicas econométricas especializadas para datos complejos."},
    {"id": "machine_learning", "name": "12. Aprendizaje Estadístico (Machine Learning)", "desc": "Modelos predictivos algorítmicos con enfoque inferencial."}
]

# Core models with full details and code templates
CORE_MODELS = {
    # Descriptive
    "arithmetic_mean": {
        "id": "arithmetic_mean",
        "name": "Media Aritmética",
        "module": "descriptive",
        "desc": "Medida de tendencia central obtenida sumando todos los valores de un conjunto de datos y dividiendo por el número total de observaciones.",
        "justification": "Indicada para resumir la localización de datos cuantitativos simétricos sin asimetría severa. Se fundamenta en el teorema del límite central para la inferencia de parámetros poblacionales.",
        "assumptions": [
            "Escala de medición intervalar o de razón.",
            "Distribución aproximadamente simétrica (sensible a valores atípicos extremos)."
        ],
        "formula": r"\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i",
        "templates": {
            "python": """import pandas as pd
df = pd.read_csv('{csv_file}')
media = df['{var}'].mean()
print(f"Media de {var}: {media}")""",
            "r": """df <- read.csv('{csv_file}')
media <- mean(df${var}, na.rm = TRUE)
print(media)""",
            "spss": """DESCRIPTIVES VARIABLES={var}
  /STATISTICS=MEAN.""",
            "stata": """import delimited "{csv_file}", clear
summarize {var}, detail""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc means data=datos mean;
   var {var};
run;"""
        }
    },
    "standard_deviation": {
        "id": "standard_deviation",
        "name": "Desviación Estándar",
        "module": "descriptive",
        "desc": "Medida de la dispersión de los datos con respecto a su media, expresada en las mismas unidades de la variable original.",
        "justification": "Permite cuantificar la variabilidad intrínseca de una muestra y es fundamental para calcular intervalos de confianza, errores estándar y para la tipificación de puntuaciones Z.",
        "assumptions": [
            "Escala de medición numérica continua o discreta.",
            "Adecuada para distribuciones unimodales."
        ],
        "formula": r"s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (X_i - \bar{X})^2}",
        "templates": {
            "python": """import pandas as pd
df = pd.read_csv('{csv_file}')
desv_est = df['{var}'].std()
print(f"Desviación Estándar de {var}: {desv_est}")""",
            "r": """df <- read.csv('{csv_file}')
desv_est <- sd(df${var}, na.rm = TRUE)
print(desv_est)""",
            "spss": """DESCRIPTIVES VARIABLES={var}
  /STATISTICS=STDDEV.""",
            "stata": """import delimited "{csv_file}", clear
summarize {var}""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc means data=datos std;
   var {var};
run;"""
        }
    },
    
    # Inference
    "independent_ttest": {
        "id": "independent_ttest",
        "name": "Prueba t de Student para Muestras Independientes",
        "module": "inference",
        "desc": "Prueba paramétrica para comparar las medias de dos grupos de observaciones independientes.",
        "justification": "Aplicable cuando se quiere contrastar si la diferencia observada entre las medias de dos poblaciones independientes es estadísticamente significativa o fruto del azar, bajo supuestos de normalidad y homocedasticidad.",
        "assumptions": [
            "Independencia de las observaciones entre y dentro de los grupos.",
            "Normalidad de la variable dependiente dentro de cada grupo.",
            "Homocedasticidad (igualdad de varianzas) poblacional."
        ],
        "formula": r"t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}, \quad s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}",
        "templates": {
            "python": """import pandas as pd
import scipy.stats as stats
df = pd.read_csv('{csv_file}')
grupo1 = df[df['{group_var}'] == '{group1}']['{dep_var}']
grupo2 = df[df['{group_var}'] == '{group2}']['{dep_var}']
t_stat, p_val = stats.ttest_ind(grupo1, grupo2, equal_var=True)
print(f"Estadístico t: {t_stat}, p-valor: {p_val}")""",
            "r": """df <- read.csv('{csv_file}')
df_filtrado <- subset(df, {group_var} %in% c('{group1}', '{group2}'))
resultado <- t.test({dep_var} ~ {group_var}, data = df_filtrado, var.equal = TRUE)
print(resultado)""",
            "spss": """T-TEST GROUPS={group_var}('{group1}' '{group2}')
  /VARIABLES={dep_var}
  /CRITERIA=CI(.95).""",
            "stata": """import delimited "{csv_file}", clear
keep if {group_var} == "{group1}" | {group_var} == "{group2}"
ttest {dep_var}, by({group_var})""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
data datos_filtrados;
   set datos;
   where {group_var} in ('{group1}', '{group2}');
run;
proc ttest data=datos_filtrados;
   class {group_var};
   var {dep_var};
run;"""
        }
    },
    "paired_ttest": {
        "id": "paired_ttest",
        "name": "Prueba t de Student Pareada (Medidas Repetidas)",
        "module": "inference",
        "desc": "Prueba paramétrica para comparar las medias de una sola muestra en dos momentos temporales o condiciones diferentes (medidas repetidas).",
        "justification": "Indicada para estudios experimentales del tipo antes/después o diseños de sujetos emparejados. Al analizar la varianza interna de las diferencias individuales, tiene mayor potencia estadística que la prueba independiente.",
        "assumptions": [
            "Los datos deben estar emparejados (cada sujeto tiene dos mediciones).",
            "La diferencia entre las dos mediciones debe seguir una distribución normal."
        ],
        "formula": r"t = \frac{\bar{D}}{s_D / \sqrt{n}}, \quad D_i = X_{1,i} - X_{2,i}",
        "templates": {
            "python": """import pandas as pd
import scipy.stats as stats
df = pd.read_csv('{csv_file}')
t_stat, p_val = stats.ttest_rel(df['{var1}'], df['{var2}'])
print(f"Estadístico t pareado: {t_stat}, p-valor: {p_val}")""",
            "r": """df <- read.csv('{csv_file}')
resultado <- t.test(df${var1}, df${var2}, paired = TRUE)
print(resultado)""",
            "spss": """T-TEST PAIRS={var1} WITH {var2} (PAIRED)
  /CRITERIA=CI(.95)
  /MISSING=ANALYSIS.""",
            "stata": """import delimited "{csv_file}", clear
ttest {var1} == {var2}""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc ttest data=datos;
   paired {var1}*{var2};
run;"""
        }
    },
    
    # Regression
    "simple_regression": {
        "id": "simple_regression",
        "name": "Regresión Lineal Simple",
        "module": "regression",
        "desc": "Modelo estadístico paramétrico que evalúa la relación lineal entre una variable predictora (X) y una variable de respuesta (Y).",
        "justification": "Se utiliza para modelar y predecir el comportamiento de una variable numérica continua en función de otra, permitiendo estimar el impacto marginal mediante la pendiente poblacional y explicar la proporción de varianza mediante el R-cuadrado.",
        "assumptions": [
            "Relación lineal entre las variables predictora y de respuesta.",
            "Independencia de los errores (residuos).",
            "Homocedasticidad (varianza constante de los residuos).",
            "Normalidad de la distribución de los residuos."
        ],
        "formula": r"Y = \beta_0 + \beta_1 X + \epsilon, \quad \beta_1 = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}, \quad \beta_0 = \bar{Y} - \beta_1\bar{X}",
        "templates": {
            "python": """import pandas as pd
import statsmodels.api as sm
df = pd.read_csv('{csv_file}')
X = sm.add_constant(df['{x_var}'])
Y = df['{y_var}']
modelo = sm.OLS(Y, X).fit()
print(modelo.summary())""",
            "r": """df <- read.csv('{csv_file}')
modelo <- lm({y_var} ~ {x_var}, data = df)
summary(modelo)""",
            "spss": """REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS R ANOVA
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT {y_var}
  /METHOD=ENTER {x_var}.""",
            "stata": """import delimited "{csv_file}", clear
regress {y_var} {x_var}""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc reg data=datos;
   model {y_var} = {x_var};
run;"""
        }
    },
    
    # Correlation & Association
    "chi_squared_test": {
        "id": "chi_squared_test",
        "name": "Prueba de Chi-cuadrado de Independencia",
        "module": "correlation",
        "desc": "Prueba no paramétrica para evaluar la asociación o independencia entre dos variables de escala categórica (nominal u ordinal).",
        "justification": "Apropiada cuando se desea determinar si la distribución conjunta de dos variables cualitativas difiere significativamente del patrón esperado si ambas variables fuesen estadísticamente independientes.",
        "assumptions": [
            "Frecuencias obtenidas mediante muestreo aleatorio.",
            "Mutua exclusividad de las categorías.",
            "Frecuencias esperadas mínimas: no más del 20% de las celdas deben tener frecuencias esperadas menores a 5, y ninguna celda debe tener frecuencia esperada menor a 1."
        ],
        "formula": r"\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \cdot C_j}{N}",
        "templates": {
            "python": """import pandas as pd
import scipy.stats as stats
df = pd.read_csv('{csv_file}')
tabla = pd.crosstab(df['{var1}'], df['{var2}'])
chi2, p_val, dof, esperados = stats.chi2_contingency(tabla)
print(f"Chi-cuadrado: {chi2}, p-valor: {p_val}")""",
            "r": """df <- read.csv('{csv_file}')
tabla <- table(df${var1}, df${var2})
resultado <- chisq.test(tabla)
print(resultado)""",
            "spss": """CROSSTABS
  /TABLES={var1} BY {var2}
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ
  /CELLS=COUNT EXPECTED.""",
            "stata": """import delimited "{csv_file}", clear
tabulate {var1} {var2}, chi2""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc freq data=datos;
   tables {var1}*{var2} / chisq;
run;"""
        }
    },
    
    # Experimental design
    "oneway_anova": {
        "id": "oneway_anova",
        "name": "ANOVA de una vía (Análisis de Varianza)",
        "module": "anova",
        "desc": "Prueba paramétrica para comparar simultáneamente las medias de tres o más grupos independientes.",
        "justification": "Se fundamenta en la descomposición de la varianza total observada en dos componentes: variabilidad intergrupo (debida al factor explicativo) y variabilidad intragrupo (error aleatorio). Evita la acumulación del error de tipo I que ocurriría al realizar múltiples comparaciones t pareadas.",
        "assumptions": [
            "La variable dependiente debe ser numérica continua.",
            "Independencia de las observaciones.",
            "Normalidad de los datos dentro de cada grupo.",
            "Homocedasticidad (igualdad de varianzas entre los grupos)."
        ],
        "formula": r"F = \frac{MS_{between}}{MS_{within}} = \frac{SS_{between} / (k-1)}{SS_{within} / (n-k)}",
        "templates": {
            "python": """import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
df = pd.read_csv('{csv_file}')
modelo = ols('{dep_var} ~ C({group_var})', data=df).fit()
tabla_anova = sm.stats.anova_lm(modelo, typ=2)
print(tabla_anova)""",
            "r": """df <- read.csv('{csv_file}')
modelo <- aov({dep_var} ~ factor({group_var}), data = df)
summary(modelo)""",
            "spss": """ONEWAY {dep_var} BY {group_var}
  /STATISTICS DESCRIPTIVES EFFECTS HOMOGENEITY
  /MISSING ANALYSIS.""",
            "stata": """import delimited "{csv_file}", clear
oneway {dep_var} {group_var}, tabulate""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc anova data=datos;
   class {group_var};
   model {dep_var} = {group_var};
run;"""
        }
    }
}

CORE_MODELS.update(LOTE_1_MODELS)

# Helper definition of other non-core base models to generate the 300+ catalog
BASE_MODELS_LIST = [
    # Descriptive
    {"id": "median_iqr", "name": "Mediana y Rango Intercuartílico (IQR)", "module": "descriptive", "desc": "Estadísticos descriptivos robustos para distribuciones asimétricas."},
    {"id": "skewness_kurtosis", "name": "Asimetría y Curtosis", "module": "descriptive", "desc": "Medidas de forma de la distribución de frecuencias."},
    {"id": "frequencies_props", "name": "Frecuencias y Proporciones", "module": "descriptive", "desc": "Recuentos y proporciones relativas para variables cualitativas."},
    
    # Probability
    {"id": "normal_dist", "name": "Distribución Normal", "module": "probability", "desc": "Distribución de probabilidad continua simétrica en forma de campana."},
    {"id": "student_t_dist", "name": "Distribución t de Student", "module": "probability", "desc": "Distribución continua con colas más anchas, útil para muestras pequeñas."},
    {"id": "chisq_dist", "name": "Distribución Chi-cuadrado", "module": "probability", "desc": "Distribución asimétrica positiva de la suma de variables normales al cuadrado."},
    {"id": "f_dist", "name": "Distribución F de Snedecor", "module": "probability", "desc": "Distribución asimétrica de la razón de dos variables chi-cuadrado independientes."},
    
    # Inference
    {"id": "z_proportion", "name": "Prueba Z para una Proporción", "module": "inference", "desc": "Prueba paramétrica para comparar una proporción observada con un valor teórico."},
    {"id": "wilcoxon_mann_whitney", "name": "Prueba U de Mann-Whitney", "module": "inference", "desc": "Alternativa no paramétrica a la prueba t independiente basada en rangos."},
    
    # Regression
    {"id": "multiple_regression", "name": "Regresión Lineal Múltiple", "module": "regression", "desc": "Modelado lineal de una variable dependiente cuantitativa con múltiples variables predictoras."},
    {"id": "binary_logistic", "name": "Regresión Logística Binaria", "module": "regression", "desc": "Modelado de la probabilidad de ocurrencia de un evento categórico binario."},
    {"id": "poisson_regression", "name": "Regresión de Poisson", "module": "regression", "desc": "Modelado para variables de conteo o tasas de ocurrencia basadas en eventos raros."},
    
    # Correlation
    {"id": "pearson_corr", "name": "Correlación de Pearson", "module": "correlation", "desc": "Medida de la fuerza y dirección de la asociación lineal entre dos variables continuas."},
    {"id": "spearman_corr", "name": "Correlación de Spearman", "module": "correlation", "desc": "Medida no paramétrica de la asociación monótona basada en rangos."},
    {"id": "kendall_corr", "name": "Correlación Tau de Kendall", "module": "correlation", "desc": "Coeficiente no paramétrico de asociación basado en la concordancia de pares."},
    
    # Multivariate
    {"id": "pca", "name": "Análisis de Componentes Principales (PCA)", "module": "multivariate", "desc": "Técnica de reducción dimensional ortogonal que maximiza la varianza explicada."},
    {"id": "efa", "name": "Análisis Factorial Exploratorio (EFA)", "module": "multivariate", "desc": "Identificación de constructos latentes explicativos a partir de variables observadas."},
    {"id": "kmeans", "name": "Agrupamiento K-Means", "module": "multivariate", "desc": "Algoritmo de particionamiento iterativo no jerárquico de observaciones en K clusters."},
    {"id": "hierarchical_cluster", "name": "Conglomerados Jerárquicos", "module": "multivariate", "desc": "Agrupamiento aglomerativo o divisivo representado gráficamente mediante un dendrograma."},
    
    # Time Series
    {"id": "ar_model", "name": "Modelo Autorregresivo (AR)", "module": "timeseries", "desc": "Modelo dependiente donde el valor actual es función lineal de valores pasados."},
    {"id": "ma_model", "name": "Modelo de Medias Móviles (MA)", "module": "timeseries", "desc": "Modelado de la dependencia temporal mediante la combinación de choques aleatorios pasados."},
    {"id": "arima", "name": "Modelo ARIMA", "module": "timeseries", "desc": "Modelado estocástico lineal integrando autorregresión, diferenciación y medias móviles."},
    {"id": "holt_winters", "name": "Suavizado Exponencial Holt-Winters", "module": "timeseries", "desc": "Algoritmo de pronóstico que suaviza tendencia y estacionalidad de forma aditiva o multiplicativa."},
    
    # ANOVA
    {"id": "twoway_anova", "name": "ANOVA de dos vías (Factorial)", "module": "anova", "desc": "Evaluación del efecto de dos variables categóricas explicativas y su interacción sobre una continua."},
    {"id": "repeated_measures_anova", "name": "ANOVA de Medias Repetidas", "module": "anova", "desc": "Extensión de ANOVA para sujetos evaluados en múltiples condiciones experimentales consecutivas."},
    {"id": "ancova", "name": "ANCOVA (Análisis de Covarianza)", "module": "anova", "desc": "Comparación de medias grupales controlando el efecto lineal de una co-variable cuantitativa continua."},
    
    # Survival
    {"id": "kaplan_meier", "name": "Estimador Kaplan-Meier", "module": "survival", "desc": "Cálculo no paramétrico de la curva de supervivencia y la probabilidad acumulada de supervivencia."},
    {"id": "cox_regression", "name": "Modelo de Riesgos Proporcionales de Cox", "module": "survival", "desc": "Modelado semi-paramétrico del impacto de covariables sobre la tasa de riesgo instantánea."},
    {"id": "logrank_test", "name": "Prueba de Log-Rank", "module": "survival", "desc": "Comparación no paramétrica de dos o más curvas de supervivencia estimadas."},
    {"id": "aft_model", "name": "Modelo de Tiempo de Falla Acelerado (AFT)", "module": "survival", "desc": "Alternativa paramétrica a la regresión de Cox que asume un efecto multiplicativo en el tiempo de supervivencia."},
    
    # SEM
    {"id": "path_analysis", "name": "Análisis de Senderos (Path Analysis)", "module": "sem", "desc": "Modelado de relaciones de causalidad directa e indirecta entre variables observadas."},
    {"id": "cfa", "name": "Análisis Factorial Confirmatorio (CFA)", "module": "sem", "desc": "Contraste riguroso de una estructura factorial teórica predefinida en variables latentes."},
    {"id": "sem_general", "name": "Modelo SEM Estructural General", "module": "sem", "desc": "Combinación de un modelo de medida (CFA) y un modelo estructural (regresiones estructurales simultáneas)."},
    {"id": "pls_sem", "name": "SEM por Mínimos Cuadrados Parciales (PLS-SEM)", "module": "sem", "desc": "Estimación basada en varianzas orientada a la predicción y exploración en modelos estructurales complejos."},
    
    # Econometrics
    {"id": "panel_data", "name": "Regresión con Datos de Panel", "module": "advanced_econometrics", "desc": "Modelado de datos combinados longitudinales y transversales con efectos fijos o aleatorios."},
    {"id": "2sls_instrumental", "name": "Variables Instrumentales (2SLS)", "module": "advanced_econometrics", "desc": "Estimación en dos etapas para corregir el sesgo de endogeneidad mediante variables instrumentales."},
    {"id": "heckman_selection", "name": "Modelo de Selección de Heckman", "module": "advanced_econometrics", "desc": "Procedimiento en dos etapas para corregir el sesgo de selección muestral no aleatorio."},
    {"id": "tobit_model", "name": "Modelo Tobit (Regresión Censurada)", "module": "advanced_econometrics", "desc": "Estimación de relaciones lineales cuando la variable dependiente está censurada por abajo o por arriba."},
    
    # Machine Learning
    {"id": "decision_tree", "name": "Árbol de Decisión Estadístico", "module": "machine_learning", "desc": "Particionado iterativo del espacio de variables mediante criterios de ganancia de información o Gini."},
    {"id": "random_forest", "name": "Bosque Aleatorio (Random Forest)", "module": "machine_learning", "desc": "Ensamble de múltiples árboles de decisión mediante agregación bootstrap y selección aleatoria de predictores."},
    {"id": "svm", "name": "Máquinas de Vectores de Soporte (SVM)", "module": "machine_learning", "desc": "Búsqueda del hiperplano de separación óptimo maximizando el margen entre clases."},
    {"id": "mlp_network", "name": "Red Neuronal Perceptrón Multicapa (MLP)", "module": "machine_learning", "desc": "Modelo de aproximación de funciones no lineales con capas ocultas y entrenamiento por propagación hacia atrás."}
]

# Variations descriptors
VARIATIONS = [
    {
        "suffix": "Estándar Frecuentista",
        "desc_add": " Estimación por máxima verosimilitud o mínimos cuadrados ordinarios según la teoría clásica asintótica.",
        "just_add": " Enfoque clásico generalizado que asume la validez asintótica de las estimaciones puntuales bajo grandes muestras.",
        "ass_add": ["Adecuado tamaño muestral para asintótica normal."],
        "adv_tag": False
    },
    {
        "suffix": "con corrección de Welch para varianzas desiguales",
        "desc_add": " Versión ajustada que calcula los grados de libertad mediante la aproximación de Satterthwaite para mitigar la heterocedasticidad.",
        "just_add": " Crucial para prevenir el incremento inaceptable del error de Tipo I cuando las muestras son asimétricas en tamaño y variabilidad.",
        "ass_add": ["No requiere igualdad de varianzas poblacionales (homocedasticidad)."],
        "adv_tag": True
    },
    {
        "suffix": "robusto con Remuestreo Bootstrap (B=10,000)",
        "desc_add": " Estimación mediante métodos no paramétricos de remuestreo iterativo para el cálculo de intervalos de confianza por percentil corregido (BCa).",
        "just_add": " Proporciona inferencia válida cuando los supuestos clásicos de distribución normal se violan severamente o la muestra es pequeña.",
        "ass_add": ["La muestra original es representativa de la población subyacente."],
        "adv_tag": True
    },
    {
        "suffix": "con Ajuste Bayesiano (Prioris Conjugadas)",
        "desc_add": " Análisis probabilístico que combina una distribución a priori teórica o empírica con la verosimilitud de la muestra para obtener una distribución a posteriori completa.",
        "just_add": " Permite incorporar conocimiento experto previo de manera formal y modela directamente la probabilidad del parámetro en lugar de tasas de error a largo plazo.",
        "ass_add": ["Especificación matemáticamente consistente de la distribución a priori."],
        "adv_tag": True
    },
    {
        "suffix": "con control de la Tasa de Falso Descubrimiento (FDR)",
        "desc_add": " Procedimiento de ajuste multivariado mediante la corrección de Benjamini-Hochberg sobre el vector de valores p obtenidos.",
        "just_add": " Imprescindible en estudios confirmatorios masivos o genómicos para equilibrar el control de falsos positivos sin sacrificar excesiva potencia estadística.",
        "ass_add": ["Independencia o dependencia positiva débil entre las pruebas de hipótesis individuales."],
        "adv_tag": True
    },
    {
        "suffix": "con corrección conservadora de Bonferroni",
        "desc_add": " Ajuste de nivel alfa crítico dividido estrictamente por la cantidad total de comparaciones simultáneas realizadas.",
        "just_add": " El método de control de error tipo I de familia más estricto y tradicional, ideal para asegurar la máxima certidumbre científica.",
        "ass_add": ["Control del nivel alfa global de la familia de comparaciones al nivel nominal especificado."],
        "adv_tag": True
    },
    {
        "suffix": "con Estimación Robusta M (Huber M-estimator)",
        "desc_add": " Estimación de parámetros ponderada iterativamente mediante una función de pérdida de Huber que minimiza la influencia de los outliers.",
        "just_add": " Evita el sesgo severo en estimaciones lineales cuando existen datos anómalos o colas de distribución pesadas sin eliminar registros manualmente.",
        "ass_add": ["Independencia en los errores y asunción de outliers de carácter no estructural."],
        "adv_tag": True
    }
]

def get_model_catalog():
    """Generates programmatically 300+ models from core and base list."""
    catalog = []
    
    # Add Core models first (7 items)
    for model_id, model_data in CORE_MODELS.items():
        catalog.append({
            "id": model_data["id"],
            "name": model_data["name"] + " (Core)",
            "module": model_data["module"],
            "desc": model_data["desc"],
            "justification": model_data["justification"],
            "assumptions": model_data["assumptions"],
            "formula": model_data["formula"],
            "templates": model_data["templates"],
            "is_core": True,
            "is_advanced": False
        })
        
    # Combine Core models keys and Base models
    all_bases = []
    for model_id, model_data in CORE_MODELS.items():
        all_bases.append({"id": model_id, "name": model_data["name"], "module": model_data["module"], "desc": model_data["desc"]})
    all_bases.extend(BASE_MODELS_LIST)
    
    # Generate variations
    # We have around 42 unique base models. We generate 7 variations for each = 294 models.
    # plus the original 7 core models = 301 models.
    for base in all_bases:
        for idx, var in enumerate(VARIATIONS):
            # Formulate the ID
            var_id = f"{base['id']}_var{idx+1}"
            
            # Formulate title
            base_name = base["name"].replace(" (Core)", "")
            var_name = f"{base_name} {var['suffix']}"
            
            # Formulate description
            var_desc = base["desc"] + var["desc_add"]
            
            # Formulate justification
            core_just = CORE_MODELS[base["id"]]["justification"] if base["id"] in CORE_MODELS else "Recomendado para análisis de nivel doctoral que requiere rigor estadístico bajo condiciones específicas del diseño de investigación."
            var_just = core_just + var["just_add"]
            
            # Formulate assumptions
            core_ass = CORE_MODELS[base["id"]]["assumptions"] if base["id"] in CORE_MODELS else ["Escala de medición e independencia metodológica coherente con el modelo base."]
            var_ass = core_ass + var["ass_add"]
            
            # Check for core templates to link, or empty for others
            templates = None
            formula = ""
            if base["id"] in CORE_MODELS:
                templates = CORE_MODELS[base["id"]]["templates"]
                formula = CORE_MODELS[base["id"]]["formula"]
            
            catalog.append({
                "id": var_id,
                "name": var_name,
                "module": base["module"],
                "desc": var_desc,
                "justification": var_just,
                "assumptions": var_ass,
                "formula": formula,
                "templates": templates,
                "is_core": False,
                "is_advanced": var["adv_tag"]
            })
            
    return catalog

def get_model_procedure(model):
    """Devuelve una guía procedimental de 5 pasos para la aplicación del modelo."""
    m_name = model.get("name", "Modelo Estadístico")
    m_mod = model.get("module", "general")
    
    return [
        {
            "step": "Paso 1: Especificación y Formulación de Hipótesis",
            "desc": f"Definir con precisión el objetivo analítico para {m_name}. Formular la Hipótesis Nula (H₀: No existe efecto / relación o los parámetros son iguales a cero) y la Hipótesis Alternativa (H₁: Existe un efecto estadísticamente relevante)."
        },
        {
            "step": "Paso 2: Verificación Previa de Supuestos Metodológicos",
            "desc": "Auditar la escala de medición de las variables y verificar los supuestos normativos requeridos: " + ", ".join(model.get("assumptions", ["Independencia de errores", "Medición adecuada"])) + "."
        },
        {
            "step": "Paso 3: Estimación y Ajuste del Modelo",
            "desc": f"Ejecutar los algoritmos de estimación (por ejemplo, Mínimos Cuadrados Ordinarios OLS, Máxima Verosimilitud ML o Métodos No Paramétricos) en el lenguaje seleccionado (Python, R, SPSS, Stata o SAS) utilizando el conjunto de datos limpio."
        },
        {
            "step": "Paso 4: Evaluación de Diagnóstico y Bondad de Ajuste",
            "desc": f"Inspeccionar el estadístico de prueba calculado, los grados de libertad y el p-valor. Comparar contra el nivel de significancia nominal (α = 0.05). Analizar residuos y la bondad de ajuste de {m_name}."
        },
        {
            "step": "Paso 5: Interpretación Práctica y Reporte Científico",
            "desc": "Traducir los coeficientes e intervalos de confianza del 95% en conclusiones sustantivas de la investigación. Documentar la justificación académica formal y guardar el script de código reproducible."
        }
    ]

def recommend_models_for_dataset(df):
    """Evalúa las características de un DataFrame y sugiere modelos relevantes de la biblioteca."""
    if df is None or df.empty:
        return []
    
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    n_rows = len(df)
    
    recommendations = []
    catalog = MODEL_CATALOG
    
    for m in catalog:
        mod = m["module"]
        m_id = m["id"]
        match_score = 50.0
        reasons = []
        
        if len(num_cols) >= 1 and mod in ["descriptive", "probability"]:
            match_score += 35
            reasons.append(f"Ideal para explorar {len(num_cols)} variables numéricas detectadas")
        elif len(num_cols) >= 2 and mod in ["regression", "correlation"]:
            match_score += 40
            reasons.append("Detectadas 2 o más variables numéricas aptas para correlación/regresión")
        elif len(num_cols) >= 1 and len(cat_cols) >= 1 and mod in ["inference", "anova"]:
            match_score += 42
            reasons.append("Detectada combinación de variable dependiente numérica y factor categórico")
        elif len(cat_cols) >= 2 and (mod == "correlation" or "chi" in m_id):
            match_score += 45
            reasons.append("Detectadas variables categóricas aptas para análisis de independencia (Chi-cuadrado)")
        elif len(num_cols) >= 3 and mod in ["multivariate", "machine_learning"]:
            match_score += 38
            reasons.append("Estructura multivariada de alta dimensión adecuada para clustering / reducción de dimensión")
            
        if m.get("is_core"):
            match_score += 10
            
        match_score = min(98.0, round(match_score, 1))
        
        if match_score >= 70.0:
            recommendations.append({
                "model": m,
                "score": match_score,
                "reasons": reasons
            })
            
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:15]

# Export items
MODEL_CATALOG = get_model_catalog()

