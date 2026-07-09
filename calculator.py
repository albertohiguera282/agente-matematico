# -*- coding: utf-8 -*-
"""
Mathematical calculations engine for statistics.
Uses scipy.stats and numpy, returns results and step-by-step LaTeX formulas.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

def calculate_descriptive(data):
    """
    Computes complete descriptive statistics for a pandas Series or list.
    """
    s = pd.Series(data).dropna()
    n = len(s)
    if n == 0:
        return {"error": "El conjunto de datos está vacío."}
        
    is_numeric = pd.to_numeric(s, errors='coerce').notnull().sum() / n >= 0.85
    
    if is_numeric:
        s_num = pd.to_numeric(s, errors='coerce').dropna()
        n_num = len(s_num)
        if n_num < 2:
            return {"error": "Se requieren al menos 2 valores válidos para análisis numérico."}
            
        mean = float(s_num.mean())
        median = float(s_num.median())
        
        # Mode
        mode_res = s_num.mode()
        mode = float(mode_res.iloc[0]) if not mode_res.empty else mean
        
        var = float(s_num.var(ddof=1))
        std = float(s_num.std(ddof=1))
        
        # Percentiles
        p25 = float(np.percentile(s_num, 25))
        p50 = float(np.percentile(s_num, 50))
        p75 = float(np.percentile(s_num, 75))
        iqr = p75 - p25
        
        # Shape
        skew = float(stats.skew(s_num, bias=False)) if n_num > 2 else 0.0
        kurt = float(stats.kurtosis(s_num, bias=False)) if n_num > 3 else 0.0
        
        # CI 95%
        se = std / np.sqrt(n_num)
        t_crit = float(stats.t.ppf(0.975, df=n_num-1))
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        
        # LaTeX step-by-step
        latex_str = rf"""
\text{{1. Cálculo de la Media Aritmética:}}
\\ \bar{{X}} = \frac{1}{{{n_num}}} \sum X_i = {mean:.4f}
\\ \text{{2. Cálculo de la Varianza Muestral y Desviación Estándar:}}
\\ s^2 = \frac{{\sum (X_i - \bar{{X}})^2}}{{{n_num} - 1}} = {var:.4f}
\\ s = \sqrt{{s^2}} = \sqrt{{{var:.4f}}} = {std:.4f}
\\ \text{{3. Intervalo de Confianza al 95\% para la Media:}}
\\ E.E. = \frac{{s}}{{\sqrt{{n}}}} = \frac{{{std:.4f}}}{{\sqrt{{{n_num}}}}} = {se:.4f}
\\ t_{{crit, 0.025}} = {t_crit:.4f} \quad (\text{{grados de libertad: }} {n_num-1})
\\ \text{{I.C. 95\%}} = \bar{{X}} \pm t_{{crit}} \cdot E.E. = {mean:.4f} \pm ({t_crit:.4f} \cdot {se:.4f})
\\ \text{{I.C. 95\%}} = [{ci_lower:.4f}, \, {ci_upper:.4f}]
"""
        return {
            "type": "numeric",
            "n": n_num,
            "mean": mean,
            "median": median,
            "mode": mode,
            "variance": var,
            "std_dev": std,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "iqr": iqr,
            "skewness": skew,
            "kurtosis": kurt,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "latex": latex_str
        }
    else:
        # Categorical
        counts = s.value_counts()
        total = len(s)
        freq_table = []
        for val, count in counts.items():
            freq_table.append({
                "value": str(val),
                "count": int(count),
                "percentage": float(count / total * 100)
            })
            
        latex_str = rf"""
\text{{Fórmula de Proporciones:}}
\\ P_i = \frac{{f_i}}{{N}} \times 100\%
\\ \text{{Total de observaciones válidas (N): }} {total}
"""
        return {
            "type": "categorical",
            "n": total,
            "table": freq_table,
            "latex": latex_str
        }

def calculate_independent_ttest(g1, g2):
    """
    Independent samples t-test (assumes equal variances, Welch t-test can be inferred).
    """
    g1 = pd.Series(g1).dropna()
    g2 = pd.Series(g2).dropna()
    n1, n2 = len(g1), len(g2)
    
    if n1 < 2 or n2 < 2:
        return {"error": "Ambos grupos deben tener al menos 2 observaciones válidas."}
        
    m1, m2 = float(g1.mean()), float(g2.mean())
    v1, v2 = float(g1.var(ddof=1)), float(g2.var(ddof=1))
    s1, s2 = float(g1.std(ddof=1)), float(g2.std(ddof=1))
    
    # Pooled variance
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    sp = np.sqrt(sp2)
    
    # t stat
    se = sp * np.sqrt(1/n1 + 1/n2)
    if se == 0:
        return {"error": "La variabilidad de los datos es cero. No se puede computar la prueba t."}
    t_stat = (m1 - m2) / se
    
    # p value
    p_val = float(stats.t.sf(np.abs(t_stat), df) * 2)
    
    # Welch t-test for comparison/alternative reference
    se_welch = np.sqrt(v1/n1 + v2/n2)
    t_welch = (m1 - m2) / se_welch if se_welch > 0 else 0
    df_welch = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)) if se_welch > 0 else 1
    p_welch = float(stats.t.sf(np.abs(t_welch), df_welch) * 2) if se_welch > 0 else 1.0
    
    latex_str = rf"""
\text{{1. Medias y Desviaciones Estándar Grupales:}}
\\ \bar{{X}}_1 = {m1:.4f}, \quad s_1 = {s1:.4f}, \quad n_1 = {n1}
\\ \bar{{X}}_2 = {m2:.4f}, \quad s_2 = {s2:.4f}, \quad n_2 = {n2}
\\ \text{{2. Varianza Agrupada (Pooled Variance) }} s_p^2:
\\ s_p^2 = \frac{{(n_1-1)s_1^2 + (n_2-1)s_2^2}}{{n_1+n_2-2}} = \frac{{({n1}-1)({v1:.4f}) + ({n2}-1)({v2:.4f})}}{{{df}}} = {sp2:.4f}
\\ s_p = \sqrt{{{sp2:.4f}}} = {sp:.4f}
\\ \text{{3. Estadístico t de Student:}}
\\ t = \frac{{\bar{{X}}_1 - \bar{{X}}_2}}{{s_p \sqrt{{\frac{{1}}{{n_1}} + \frac{{1}}{{n_2}}}}}} = \frac{{{m1:.4f} - {m2:.4f}}}{{{sp:.4f} \sqrt{{\frac{{1}}{{{n1}}} + \frac{{1}}{{{n2}}}}}}} = {t_stat:.4f}
\\ \text{{4. Grados de libertad y p-valor (dos colas):}}
\\ gl = {df}, \quad p = {p_val:.6f}
"""
    return {
        "n1": n1, "n2": n2,
        "mean1": m1, "mean2": m2,
        "sd1": s1, "sd2": s2,
        "df": df,
        "t_statistic": t_stat,
        "p_value": p_val,
        "welch_t": t_welch,
        "welch_df": df_welch,
        "welch_p": p_welch,
        "latex": latex_str
    }

def calculate_paired_ttest(v1, v2):
    """
    Paired samples t-test.
    """
    v1 = pd.Series(v1)
    v2 = pd.Series(v2)
    # Align and drop missing
    df_aligned = pd.DataFrame({"v1": v1, "v2": v2}).dropna()
    
    n = len(df_aligned)
    if n < 2:
        return {"error": "Se requieren al menos 2 observaciones completas emparejadas."}
        
    x1 = df_aligned["v1"].values
    x2 = df_aligned["v2"].values
    
    diffs = x1 - x2
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1))
    var_diff = float(diffs.var(ddof=1))
    
    se = std_diff / np.sqrt(n)
    if se == 0:
        return {"error": "La variabilidad de las diferencias es cero. No se puede calcular t."}
        
    t_stat = mean_diff / se
    df = n - 1
    p_val = float(stats.t.sf(np.abs(t_stat), df) * 2)
    
    latex_str = rf"""
\text{{1. Diferencias Individuales (D = X_1 - X_2):}}
\\ \bar{{D}} = {mean_diff:.4f}, \quad s_D = {std_diff:.4f}, \quad n = {n}
\\ \text{{2. Error Estándar de la Media de Diferencias (E.E.):}}
\\ E.E. = \frac{{s_D}}{{\sqrt{{n}}}} = \frac{{{std_diff:.4f}}}{{\sqrt{{{n}}}}} = {se:.4f}
\\ \text{{3. Estadístico t Pareado:}}
\\ t = \frac{{\bar{{D}}}}{{E.E.}} = \frac{{{mean_diff:.4f}}}{{{se:.4f}}} = {t_stat:.4f}
\\ \text{{4. Grados de libertad y p-valor (dos colas):}}
\\ gl = {df}, \quad p = {p_val:.6f}
"""
    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "df": df,
        "t_statistic": t_stat,
        "p_value": p_val,
        "latex": latex_str,
        "diffs": diffs.tolist()
    }

def calculate_chi_squared(v1, v2):
    """
    Chi-squared test of independence.
    """
    v1 = pd.Series(v1)
    v2 = pd.Series(v2)
    df_aligned = pd.DataFrame({"v1": v1, "v2": v2}).dropna()
    
    n = len(df_aligned)
    if n == 0:
        return {"error": "El dataset alineado está vacío."}
        
    # Build contingency table
    contingency = pd.crosstab(df_aligned["v1"], df_aligned["v2"])
    
    observed = contingency.values
    rows, cols = observed.shape
    if rows < 2 or cols < 2:
        return {"error": "Ambas variables categóricas deben tener al menos 2 categorías distintas."}
        
    # SciPy call
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    # Calculate categories names
    row_labels = [str(x) for x in contingency.index]
    col_labels = [str(x) for x in contingency.columns]
    
    # Build observed & expected tables for response
    obs_list = contingency.values.tolist()
    exp_list = expected.tolist()
    
    # LaTeX calculation steps
    # We show the sum for the first few terms to keep it clean
    calc_terms = []
    term_count = 0
    for r in range(rows):
        for c in range(cols):
            o = obs_list[r][c]
            e = exp_list[r][c]
            diff2 = (o - e)**2
            term = diff2 / e
            if term_count < 4:
                calc_terms.append(rf"\frac{{({o} - {e:.2f})^2}}{{{e:.2f}}} = {term:.4f}")
            term_count += 1
            
    terms_str = " + ".join(calc_terms)
    if term_count > 4:
        terms_str += " + \dots"
        
    latex_str = rf"""
\text{{1. Frecuencias Esperadas (E):}}
\\ E_{{ij}} = \frac{{\text{{Total Fila}} \times \text{{Total Columna}}}}{{N}}
\\ \text{{2. Cálculo del Estadístico Chi-cuadrado:}}
\\ \chi^2 = \sum \frac{{(O_{{ij}} - E_{{ij}})^2}}{{E_{{ij}}}}
\\ \chi^2 = {terms_str} = {chi2:.4f}
\\ \text{{3. Parámetros de la Distribución:}}
\\ gl = (r - 1)(c - 1) = ({rows} - 1)({cols} - 1) = {dof}
\\ p = {p_val:.6f}
"""
    return {
        "n": n,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "observed": obs_list,
        "expected": exp_list,
        "df": dof,
        "chi2_statistic": chi2,
        "p_value": p_val,
        "latex": latex_str
    }

def calculate_simple_regression(x, y):
    """
    Simple linear regression.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    df_aligned = pd.DataFrame({"x": x, "y": y}).dropna()
    
    n = len(df_aligned)
    if n < 3:
        return {"error": "Se requieren al menos 3 observaciones completas para ajustar una regresión."}
        
    x_val = df_aligned["x"].values
    y_val = df_aligned["y"].values
    
    # Calculate regression using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_val, y_val)
    r2 = r_value ** 2
    
    # Mean calculations
    mean_x = float(x_val.mean())
    mean_y = float(y_val.mean())
    
    # ANOVA calculations
    y_pred = intercept + slope * x_val
    ssr = float(np.sum((y_pred - mean_y) ** 2)) # Sum of squares regression
    sse = float(np.sum((y_val - y_pred) ** 2))   # Sum of squares residuals
    sst = ssr + sse                              # Total sum of squares
    
    df_reg = 1
    df_res = n - 2
    df_tot = n - 1
    
    msr = ssr / df_reg
    mse = sse / df_res if df_res > 0 else 0
    
    f_stat = msr / mse if mse > 0 else 0
    f_p_val = float(stats.f.sf(f_stat, df_reg, df_res)) if mse > 0 else 1.0
    
    latex_str = rf"""
\text{{1. Medias de las variables:}}
\\ \bar{{X}} = {mean_x:.4f}, \quad \bar{{Y}} = {mean_y:.4f}, \quad n = {n}
\\ \text{{2. Coeficientes del Modelo (Y = \beta_0 + \beta_1 X):}}
\\ \beta_1 = \frac{{\sum(X_i - \bar{{X}})(Y_i - \bar{{Y}})}}{{\sum(X_i - \bar{{X}})^2}} = {slope:.4f}
\\ \beta_0 = \bar{{Y}} - \beta_1\bar{{X}} = {mean_y:.4f} - ({slope:.4f} \cdot {mean_x:.4f}) = {intercept:.4f}
\\ \text{{3. Coeficiente de Determinación (Bondad de Ajuste):}}
\\ R = {r_value:.4f}, \quad R^2 = {r2:.4f}
\\ \text{{4. Contraste de ANOVA de la Regresión:}}
\\ F_{{calc}} = \frac{{MSR}}{{MSE}} = \frac{{{msr:.4f}}}{{{mse:.4f}}} = {f_stat:.4f} \quad (\text{{grados de libertad: 1, }} {df_res})
\\ p = {f_p_val:.6f}
"""
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r2,
        "p_value": p_value,
        "std_err": std_err,
        "anova": {
            "ssr": ssr, "sse": sse, "sst": sst,
            "df_reg": df_reg, "df_res": df_res, "df_tot": df_tot,
            "msr": msr, "mse": mse,
            "f_statistic": f_stat,
            "p_value": f_p_val
        },
        "latex": latex_str
    }

def calculate_oneway_anova(groups, values):
    """
    One-way Analysis of Variance (ANOVA) for 3+ groups.
    """
    groups = pd.Series(groups)
    values = pd.Series(values)
    df_aligned = pd.DataFrame({"group": groups, "value": values}).dropna()
    
    unique_groups = df_aligned["group"].unique()
    k = len(unique_groups)
    if k < 3:
        return {"error": "El factor explicativo debe tener al menos 3 categorías distintas para ANOVA."}
        
    grouped_data = [df_aligned[df_aligned["group"] == g]["value"].values for g in unique_groups]
    
    # Total sample size
    N = len(df_aligned)
    if any(len(g) < 2 for g in grouped_data):
        return {"error": "Cada grupo debe tener al menos 2 observaciones válidas."}
        
    # Total mean
    grand_mean = float(df_aligned["value"].mean())
    
    # SS Between and SS Within
    ss_between = 0.0
    ss_within = 0.0
    
    group_stats = []
    
    for g, data in zip(unique_groups, grouped_data):
        n_g = len(data)
        mean_g = float(data.mean())
        var_g = float(data.var(ddof=1))
        std_g = float(data.std(ddof=1))
        
        ss_between += n_g * (mean_g - grand_mean) ** 2
        ss_within += (n_g - 1) * var_g
        
        group_stats.append({
            "group": str(g),
            "n": n_g,
            "mean": mean_g,
            "sd": std_g
        })
        
    sst = ss_between + ss_within
    
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    f_stat = ms_between / ms_within
    p_val = float(stats.f.sf(f_stat, df_between, df_within))
    
    latex_str = rf"""
\text{{1. Media Global y Totales:}}
\\ \bar{{X}}_{{grand}} = {grand_mean:.4f}, \quad N = {N}, \quad k = {k} \text{{ grupos}}
\\ \text{{2. Descomposición de Suma de Cuadrados (SS):}}
\\ SS_{{between}} = \sum n_g (\bar{{X}}_g - \bar{{X}}_{{grand}})^2 = {ss_between:.4f}
\\ SS_{{within}} = \sum (n_g - 1)s_g^2 = {ss_within:.4f}
\\ SST = SS_{{between}} + SS_{{within}} = {sst:.4f}
\\ \text{{3. Medias Cuadráticas (MS) y Estadístico F:}}
\\ MS_{{between}} = \frac{{SS_{{between}}}}{{k - 1}} = \frac{{{ss_between:.4f}}}{{{df_between}}} = {ms_between:.4f}
\\ MS_{{within}} = \frac{{SS_{{within}}}}{{N - k}} = \frac{{{ss_within:.4f}}}{{{df_within}}} = {ms_within:.4f}
\\ F = \frac{{MS_{{between}}}}{{MS_{{within}}}} = \frac{{{ms_between:.4f}}}{{{ms_within:.4f}}} = {f_stat:.4f}
\\ \text{{4. Grados de libertad y p-valor:}}
\\ gl_{{between}} = {df_between}, \quad gl_{{within}} = {df_within}, \quad p = {p_val:.6f}
"""
    return {
        "N": N, "k": k,
        "grand_mean": grand_mean,
        "groups": group_stats,
        "ss_between": ss_between, "ss_within": ss_within, "sst": sst,
        "df_between": df_between, "df_within": df_within, "df_total": df_total,
        "ms_between": ms_between, "ms_within": ms_within,
        "f_statistic": f_stat,
        "p_value": p_val,
        "latex": latex_str
    }
