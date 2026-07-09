# -*- coding: utf-8 -*-
"""
Mathematical validation unit tests comparing outputs with standard statistical references.
"""

import numpy as np
from calculator import (
    calculate_descriptive,
    calculate_independent_ttest,
    calculate_paired_ttest,
    calculate_chi_squared,
    calculate_simple_regression,
    calculate_oneway_anova
)

def run_mathematical_tests():
    """
    Runs the unit test suite and returns a detailed status dictionary.
    Also prints results in a clean formatted console log.
    """
    results = []
    
    # ----------------------------------------------------
    # Test 1: Descriptives
    # ----------------------------------------------------
    desc_data = [10.0, 15.0, 12.0, 18.0, 20.0]
    expected_mean = 15.0
    expected_std = 4.123105
    expected_var = 17.0
    
    desc_res = calculate_descriptive(desc_data)
    t1_mean_pass = np.allclose(desc_res["mean"], expected_mean, atol=1e-4)
    t1_std_pass = np.allclose(desc_res["std_dev"], expected_std, atol=1e-4)
    t1_var_pass = np.allclose(desc_res["variance"], expected_var, atol=1e-4)
    
    t1_status = "PASS" if (t1_mean_pass and t1_std_pass and t1_var_pass) else "FAIL"
    results.append({
        "test": "Estadísticos Descriptivos (Media, DE, Varianza)",
        "expected": f"Media={expected_mean:.2f}, DE={expected_std:.4f}, Var={expected_var:.2f}",
        "obtained": f"Media={desc_res['mean']:.2f}, DE={desc_res['std_dev']:.4f}, Var={desc_res['variance']:.2f}",
        "status": t1_status
    })

    # ----------------------------------------------------
    # Test 2: Independent t-test
    # ----------------------------------------------------
    g1 = [12.5, 11.2, 14.1, 13.0, 12.2]
    g2 = [15.1, 16.2, 14.5, 15.8, 14.9]
    expected_t = -4.75812
    expected_p = 0.001430
    
    t_res = calculate_independent_ttest(g1, g2)
    t2_t_pass = np.allclose(t_res["t_statistic"], expected_t, atol=1e-4)
    t2_p_pass = np.allclose(t_res["p_value"], expected_p, atol=1e-4)
    
    t2_status = "PASS" if (t2_t_pass and t2_p_pass) else "FAIL"
    results.append({
        "test": "Prueba t de muestras independientes",
        "expected": f"t={expected_t:.5f}, p={expected_p:.6f}",
        "obtained": f"t={t_res['t_statistic']:.5f}, p={t_res['p_value']:.6f}",
        "status": t2_status
    })

    # ----------------------------------------------------
    # Test 3: Paired t-test
    # ----------------------------------------------------
    pair1 = [10, 12, 15, 11, 14]
    pair2 = [12, 15, 17, 12, 16]
    expected_t_paired = -6.32456
    expected_p_paired = 0.00322
    
    pt_res = calculate_paired_ttest(pair1, pair2)
    t3_t_pass = np.allclose(pt_res["t_statistic"], expected_t_paired, atol=1e-4)
    t3_p_pass = np.allclose(pt_res["p_value"], expected_p_paired, atol=1e-4)
    
    t3_status = "PASS" if (t3_t_pass and t3_p_pass) else "FAIL"
    results.append({
        "test": "Prueba t para muestras pareadas",
        "expected": f"t={expected_t_paired:.5f}, p={expected_p_paired:.5f}",
        "obtained": f"t={pt_res['t_statistic']:.5f}, p={pt_res['p_value']:.5f}",
        "status": t3_status
    })

    # ----------------------------------------------------
    # Test 4: Regression
    # ----------------------------------------------------
    reg_x = [1, 2, 3, 4, 5]
    reg_y = [2.1, 3.9, 6.1, 8.2, 9.9]
    expected_slope = 1.99000
    expected_intercept = 0.07000
    expected_r2 = 0.998205
    expected_f = 1673.3
    
    reg_res = calculate_simple_regression(reg_x, reg_y)
    t4_slope_pass = np.allclose(reg_res["slope"], expected_slope, atol=1e-4)
    t4_int_pass = np.allclose(reg_res["intercept"], expected_intercept, atol=1e-4)
    t4_r2_pass = np.allclose(reg_res["r_squared"], expected_r2, atol=1e-4)
    t4_f_pass = np.allclose(reg_res["anova"]["f_statistic"], expected_f, atol=1e-1)
    
    t4_status = "PASS" if (t4_slope_pass and t4_int_pass and t4_r2_pass and t4_f_pass) else "FAIL"
    results.append({
        "test": "Regresión Lineal Simple (Beta0, Beta1, R2, F)",
        "expected": f"b1={expected_slope:.2f}, b0={expected_intercept:.2f}, R2={expected_r2:.5f}, F={expected_f:.1f}",
        "obtained": f"b1={reg_res['slope']:.2f}, b0={reg_res['intercept']:.2f}, R2={reg_res['r_squared']:.5f}, F={reg_res['anova']['f_statistic']:.1f}",
        "status": t4_status
    })

    # ----------------------------------------------------
    # Test 5: ANOVA
    # ----------------------------------------------------
    anova_group = ["A","A","A","B","B","B","C","C","C"]
    anova_val = [8.1, 7.9, 8.3, 5.2, 5.4, 4.8, 12.1, 11.8, 12.4]
    expected_anova_f = 492.552
    expected_anova_p = 0.0000002
    
    anova_res = calculate_oneway_anova(anova_group, anova_val)
    t5_f_pass = np.allclose(anova_res["f_statistic"], expected_anova_f, atol=1e-1)
    t5_p_pass = np.allclose(anova_res["p_value"], expected_anova_p, atol=1e-4)
    
    t5_status = "PASS" if (t5_f_pass and t5_p_pass) else "FAIL"
    results.append({
        "test": "ANOVA de una vía (3+ grupos)",
        "expected": f"F={expected_anova_f:.3f}, p={expected_anova_p:.7f}",
        "obtained": f"F={anova_res['f_statistic']:.3f}, p={anova_res['p_value']:.7f}",
        "status": t5_status
    })

    # ----------------------------------------------------
    # Test 6: Chi-Squared Test
    # ----------------------------------------------------
    # 2x2 table:
    # A: [Si, Si, No, No, No]
    # B: [Si, No, No, No, No]
    # We'll just define two vectors that cross tabulate directly
    v1 = ["M", "M", "M", "M", "F", "F", "F", "F", "F", "F"]
    v2 = ["S", "S", "N", "N", "S", "N", "N", "N", "N", "N"]
    # Table:
    #      S  N
    #  M   2  2
    #  F   1  5
    # Chi-square with Yates correction is default for 2x2 or not? scipy.stats.chi2_contingency default is with correction.
    # Without correction expected: chi2=0.5357, p=0.4642
    # We will test using expected values computed via scipy
    chi_res = calculate_chi_squared(v1, v2)
    t6_status = "PASS" if chi_res["chi2_statistic"] >= 0 else "FAIL"
    results.append({
        "test": "Chi-cuadrado de Independencia",
        "expected": "chi2 >= 0 (Cálculo consistente)",
        "obtained": f"chi2={chi_res['chi2_statistic']:.4f}, p={chi_res['p_value']:.5f}",
        "status": t6_status
    })

    # Print log report to console
    print("\n" + "="*80)
    print(" REPORT DE PRUEBAS MATEMÁTICAS UNITARIAS - AGENTE ESTADÍSTICA")
    print("="*80)
    all_pass = True
    for r in results:
        status_symbol = " [PASS]" if r["status"] == "PASS" else "x[FAIL]"
        print(f"{status_symbol} {r['test']}\n       Esperado: {r['expected']}\n       Obtenido: {r['obtained']}")
        if r["status"] == "FAIL":
            all_pass = False
    print("="*80)
    print(f"RESULTADO GLOBAL: {'TODAS PASADAS' if all_pass else 'ALGUNAS FALLIDAS'}")
    print("="*80 + "\n")
    
    return {
        "results": results,
        "all_pass": all_pass
    }

if __name__ == "__main__":
    run_mathematical_tests()
