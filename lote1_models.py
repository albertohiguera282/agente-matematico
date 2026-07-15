# -*- coding: utf-8 -*-
"""
LOTE 1 — Modelos base a integrar en CORE_MODELS (database.py)
Descriptiva + Inferencia: median_iqr, skewness_kurtosis, frequencies_props,
z_proportion, wilcoxon_mann_whitney
"""

LOTE_1_MODELS = {

    "median_iqr": {
        "id": "median_iqr",
        "name": "Mediana y Rango Intercuartílico (IQR)",
        "module": "descriptive",
        "desc": "Estadísticos descriptivos robustos de tendencia central y dispersión, basados en percentiles y resistentes a valores atípicos.",
        "justification": "Indicado cuando la distribución de los datos presenta asimetría severa o valores atípicos extremos que distorsionarían la media y la desviación estándar. La mediana y el IQR son estadísticos de orden, insensibles a la magnitud de los valores extremos.",
        "assumptions": [
            "Escala de medición al menos ordinal.",
            "No requiere supuesto de normalidad ni de simetría.",
            "Recomendado especialmente cuando existen outliers o la muestra es pequeña."
        ],
        "formula": r"\text{Mediana} = X_{\left(\frac{n+1}{2}\right)}, \quad IQR = Q_3 - Q_1",
        "templates": {
            "python": """import pandas as pd
df = pd.read_csv('{csv_file}')
mediana = df['{var}'].median()
q1 = df['{var}'].quantile(0.25)
q3 = df['{var}'].quantile(0.75)
iqr = q3 - q1
print(f"Mediana de {var}: {mediana}")
print(f"IQR de {var}: {iqr} (Q1={q1}, Q3={q3})")""",
            "r": """df <- read.csv('{csv_file}')
mediana <- median(df${var}, na.rm = TRUE)
q1 <- quantile(df${var}, 0.25, na.rm = TRUE)
q3 <- quantile(df${var}, 0.75, na.rm = TRUE)
iqr_val <- IQR(df${var}, na.rm = TRUE)
print(list(mediana = mediana, IQR = iqr_val, Q1 = q1, Q3 = q3))""",
            "spss": """EXAMINE VARIABLES={var}
  /STATISTICS=EXTREME
  /PERCENTILES(25,50,75) HAVERAGE.""",
            "stata": """import delimited "{csv_file}", clear
summarize {var}, detail
* La mediana y percentiles 25/75 aparecen en el detalle""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc means data=datos median q1 q3 qrange;
   var {var};
run;"""
        }
    },

    "skewness_kurtosis": {
        "id": "skewness_kurtosis",
        "name": "Asimetría y Curtosis",
        "module": "descriptive",
        "desc": "Medidas de la forma de la distribución: asimetría (dirección de la cola) y curtosis (concentración/apuntamiento respecto a la normal).",
        "justification": "Fundamental como diagnóstico previo a la selección de pruebas paramétricas. Valores de asimetría fuera de [-1, 1] o curtosis fuera de [-2, 2] sugieren desviaciones relevantes de la normalidad que pueden invalidar supuestos de pruebas t, ANOVA o regresión.",
        "assumptions": [
            "Variable de escala intervalar o de razón.",
            "Requiere un tamaño muestral razonable (n > 30) para estimaciones estables."
        ],
        "formula": r"Asimetría = \frac{n}{(n-1)(n-2)} \sum \left(\frac{X_i - \bar{X}}{s}\right)^3, \quad Curtosis = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum \left(\frac{X_i - \bar{X}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}",
        "templates": {
            "python": """import pandas as pd
from scipy import stats
df = pd.read_csv('{csv_file}')
asimetria = stats.skew(df['{var}'].dropna(), bias=False)
curtosis = stats.kurtosis(df['{var}'].dropna(), bias=False)
print(f"Asimetría de {var}: {asimetria:.4f}")
print(f"Curtosis (exceso) de {var}: {curtosis:.4f}")""",
            "r": """df <- read.csv('{csv_file}')
library(moments)
asimetria <- skewness(df${var}, na.rm = TRUE)
curtosis <- kurtosis(df${var}, na.rm = TRUE) - 3
print(list(asimetria = asimetria, curtosis_exceso = curtosis))""",
            "spss": """FREQUENCIES VARIABLES={var}
  /STATISTICS=SKEWNESS SEKEWNESS KURTOSIS SEKURTOSIS.""",
            "stata": """import delimited "{csv_file}", clear
summarize {var}, detail
* Skewness y Kurtosis aparecen en el resumen detallado""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc means data=datos skewness kurtosis;
   var {var};
run;"""
        }
    },

    "frequencies_props": {
        "id": "frequencies_props",
        "name": "Frecuencias y Proporciones",
        "module": "descriptive",
        "desc": "Recuento y proporción relativa de categorías para variables cualitativas nominales u ordinales, base de cualquier análisis exploratorio categórico.",
        "justification": "Constituye el primer paso obligatorio en el análisis de cualquier variable cualitativa, permitiendo caracterizar la distribución de la muestra antes de aplicar pruebas de asociación o independencia como Chi-cuadrado.",
        "assumptions": [
            "Variable de escala nominal u ordinal.",
            "Las categorías deben ser mutuamente excluyentes y exhaustivas."
        ],
        "formula": r"P_i = \frac{f_i}{N} \times 100\%",
        "templates": {
            "python": """import pandas as pd
df = pd.read_csv('{csv_file}')
tabla = df['{var}'].value_counts()
proporciones = df['{var}'].value_counts(normalize=True) * 100
resultado = pd.DataFrame({{'Frecuencia': tabla, 'Porcentaje': proporciones}})
print(resultado)""",
            "r": """df <- read.csv('{csv_file}')
tabla <- table(df${var})
proporciones <- prop.table(tabla) * 100
print(tabla)
print(round(proporciones, 2))""",
            "spss": """FREQUENCIES VARIABLES={var}
  /ORDER=ANALYSIS.""",
            "stata": """import delimited "{csv_file}", clear
tabulate {var}""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc freq data=datos;
   tables {var};
run;"""
        }
    },

    "z_proportion": {
        "id": "z_proportion",
        "name": "Prueba Z para una Proporción",
        "module": "inference",
        "desc": "Prueba paramétrica que contrasta si una proporción muestral observada difiere significativamente de un valor teórico o hipotético de referencia.",
        "justification": "Aplicable cuando se desea validar si la tasa de ocurrencia de un evento binario en la muestra (ej. tasa de aprobación, prevalencia) coincide con un estándar poblacional conocido, bajo el supuesto de aproximación normal de la proporción muestral (np y n(1-p) suficientemente grandes).",
        "assumptions": [
            "Muestra aleatoria simple.",
            "np₀ ≥ 5 y n(1-p₀) ≥ 5 para garantizar la aproximación normal.",
            "La variable de interés es dicotómica (éxito/fracaso)."
        ],
        "formula": r"Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}",
        "templates": {
            "python": """import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
df = pd.read_csv('{csv_file}')
exitos = (df['{var}'] == '{success_value}').sum()
n = df['{var}'].notna().sum()
p0 = {p0}
z_stat, p_val = proportions_ztest(count=exitos, nobs=n, value=p0)
print(f"Estadístico Z: {{z_stat:.4f}}, p-valor: {{p_val:.6f}}")""",
            "r": """df <- read.csv('{csv_file}')
exitos <- sum(df${var} == '{success_value}', na.rm = TRUE)
n <- sum(!is.na(df${var}))
resultado <- prop.test(exitos, n, p = {p0}, correct = FALSE)
print(resultado)""",
            "spss": """* SPSS no tiene un comando directo; usar sintaxis NPTESTS
NPTESTS
  /ONESAMPLE TEST (${var}) BINOMIAL(TESTVALUE={p0}).""",
            "stata": """import delimited "{csv_file}", clear
prtest {var} == {p0}""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc freq data=datos;
   tables {var} / binomial(p={p0});
run;"""
        }
    },

    "wilcoxon_mann_whitney": {
        "id": "wilcoxon_mann_whitney",
        "name": "Prueba U de Mann-Whitney",
        "module": "inference",
        "desc": "Prueba no paramétrica basada en rangos para comparar la tendencia central de dos grupos independientes cuando no se cumple el supuesto de normalidad.",
        "justification": "Es la alternativa no paramétrica estándar a la prueba t de Student para muestras independientes. Se utiliza cuando la variable dependiente es ordinal o cuando la distribución de los datos numéricos se aparta severamente de la normalidad, sin requerir dicho supuesto.",
        "assumptions": [
            "Independencia de las observaciones entre y dentro de los grupos.",
            "La variable dependiente es al menos ordinal.",
            "Las distribuciones de ambos grupos tienen forma similar (para interpretar el resultado como comparación de medianas)."
        ],
        "formula": r"U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1, \quad U = \min(U_1, U_2)",
        "templates": {
            "python": """import pandas as pd
import scipy.stats as stats
df = pd.read_csv('{csv_file}')
grupo1 = df[df['{group_var}'] == '{group1}']['{dep_var}']
grupo2 = df[df['{group_var}'] == '{group2}']['{dep_var}']
u_stat, p_val = stats.mannwhitneyu(grupo1, grupo2, alternative='two-sided')
print(f"Estadístico U: {{u_stat}}, p-valor: {{p_val:.6f}}")""",
            "r": """df <- read.csv('{csv_file}')
df_filtrado <- subset(df, {group_var} %in% c('{group1}', '{group2}'))
resultado <- wilcox.test({dep_var} ~ {group_var}, data = df_filtrado)
print(resultado)""",
            "spss": """NPAR TESTS
  /M-W= {dep_var} BY {group_var}('{group1}' '{group2}')
  /MISSING ANALYSIS.""",
            "stata": """import delimited "{csv_file}", clear
keep if {group_var} == "{group1}" | {group_var} == "{group2}"
ranksum {dep_var}, by({group_var})""",
            "sas": """proc import datafile="{csv_file}" out=datos dbms=csv replace; run;
proc npar1way data=datos wilcoxon;
   class {group_var};
   var {dep_var};
run;"""
        }
    }
}