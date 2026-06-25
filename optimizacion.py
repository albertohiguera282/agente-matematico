import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linprog
import pandas as pd


# ─────────────────────────────────────────────
# 1. PROGRAMACIÓN LINEAL GRÁFICA (2 variables)
# ─────────────────────────────────────────────
def resolver_programacion_lineal_grafico(c_obj, restricciones, tipo_opt="max"):
    """
    Resuelve gráficamente un PL de 2 variables.
    c_obj        : (c1, c2)  coeficientes de la FO
    restricciones: lista de (a, b, rhs)  → a·x + b·y ≤ rhs
    tipo_opt     : 'max' o 'min'
    """
    c1, c2 = c_obj

    lineas = list(restricciones) + [(1, 0, 0), (0, 1, 0)]
    puntos_interseccion = []

    for i in range(len(lineas)):
        for j in range(i + 1, len(lineas)):
            a1, b1, r1 = lineas[i]
            a2, b2, r2 = lineas[j]
            A = np.array([[a1, b1], [a2, b2]], dtype=float)
            B = np.array([r1, r2], dtype=float)
            if abs(np.linalg.det(A)) > 1e-9:
                pt = np.linalg.solve(A, B)
                puntos_interseccion.append(pt)

    puntos_factibles = []
    for pt in puntos_interseccion:
        x, y = pt
        if x < -1e-5 or y < -1e-5:
            continue
        if all(a * x + b * y <= rhs + 1e-5 for a, b, rhs in restricciones):
            if not any(np.linalg.norm(pt - pf) < 1e-6 for pf in puntos_factibles):
                puntos_factibles.append(pt)

    if not puntos_factibles:
        raise ValueError("La región factible está vacía o no acotada.")

    valores_z = [c1 * p[0] + c2 * p[1] for p in puntos_factibles]
    idx_opt = np.argmax(valores_z) if tipo_opt == "max" else np.argmin(valores_z)
    pt_opt  = puntos_factibles[idx_opt]
    z_opt   = valores_z[idx_opt]

    # Gráfico
    fig, ax = plt.subplots(figsize=(7, 6))
    max_x = max(p[0] for p in puntos_factibles)
    max_y = max(p[1] for p in puntos_factibles)
    lx = max(max_x * 1.6, 5.0)
    ly = max(max_y * 1.6, 5.0)
    xv = np.linspace(0, lx, 300)

    colors = ["#00BFFF", "#FF6B6B", "#FFD700", "#7CFC00", "#FF69B4", "#DA70D6"]
    for k, (a, b, rhs) in enumerate(restricciones):
        col = colors[k % len(colors)]
        if b != 0:
            yv = (rhs - a * xv) / b
            ax.plot(xv, yv, color=col, lw=1.8, label=f"${a}x_1+{b}x_2\\leq{rhs}$")
        else:
            ax.axvline(rhs / a, color=col, lw=1.8, label=f"${a}x_1\\leq{rhs}$")

    if len(puntos_factibles) >= 3:
        cx = np.mean([p[0] for p in puntos_factibles])
        cy = np.mean([p[1] for p in puntos_factibles])
        pts_ord = sorted(puntos_factibles, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        poly = np.array(pts_ord)
        ax.fill(poly[:, 0], poly[:, 1], color="#00F5D4", alpha=0.25, label="Región Factible")

    for pt in puntos_factibles:
        ax.scatter(*pt, color="white", s=50, zorder=5)

    ax.scatter(*pt_opt, color="#F15BB5", s=140, zorder=6, edgecolor="black",
               label=f"Óptimo Z={z_opt:.4f}")
    ax.text(pt_opt[0], pt_opt[1] + ly * 0.03,
            f"  ({pt_opt[0]:.3f}, {pt_opt[1]:.3f})", color="white", fontweight="bold")

    ax.set_xlim(0, lx);  ax.set_ylim(0, ly)
    ax.set_xlabel("$x_1$", color="white");  ax.set_ylabel("$x_2$", color="white")
    ax.grid(color="#333333", linestyle=":", alpha=0.5)
    ax.axhline(0, color="white", lw=1); ax.axvline(0, color="white", lw=1)
    ax.legend(facecolor="#1e1e24", edgecolor="#9B5DE5", labelcolor="white", fontsize=8)
    ax.set_title(f"Solución Gráfica — {tipo_opt.upper()}", color="white")
    fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#1E1E24")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#333333")

    tabla = [{"Vértice (x1, x2)": f"({p[0]:.4f}, {p[1]:.4f})",
              "Valor Z": round(z, 4)}
             for p, z in zip(puntos_factibles, valores_z)]

    return {"Punto_Optimo": pt_opt, "Z_Optimo": z_opt,
            "Vertices_Evaluados": tabla, "Fig": fig}


# ─────────────────────────────────────────────
# 2. PROGRAMACIÓN LINEAL SIMPLEX (N variables)
# ─────────────────────────────────────────────
def resolver_programacion_lineal_simplex(funcion_objetivo, restricciones,
                                         limites, tipo_opt="max"):
    """
    Maximiza / minimiza  c·x  sujeto a  A·x ≤ b,  x ≥ 0.
    funcion_objetivo : lista de coeficientes  [c1, c2, ..., cn]
    restricciones    : matriz  [[a11,a12,...], [a21,a22,...], ...]
    limites          : lado derecho  [b1, b2, ...]
    tipo_opt         : 'max' o 'min'
    """
    c = np.array(funcion_objetivo, dtype=float)
    if tipo_opt == "max":
        c = -c

    res = linprog(c, A_ub=np.array(restricciones, dtype=float),
                  b_ub=np.array(limites, dtype=float),
                  bounds=[(0, None)] * len(funcion_objetivo), method="highs")

    if not res.success:
        return {"Estado": "Sin solución", "Mensaje": res.message}

    z = -res.fun if tipo_opt == "max" else res.fun
    variables = {f"X{i+1}": round(v, 6) for i, v in enumerate(res.x)}

    return {"Estado": "Óptimo", "Valor_Optimo": round(z, 6),
            "Variables": variables, "Mensaje": res.message}


# ─────────────────────────────────────────────────────────────
# 3. PROGRAMACIÓN LINEAL ENTERA (N variables, Branch & Bound)
# ─────────────────────────────────────────────────────────────
def resolver_programacion_lineal_entera(funcion_objetivo, restricciones,
                                        limites, variables_enteras=None,
                                        tipo_opt="max"):
    """
    PL Entera / Mixta.
    variables_enteras: índices (base-0) de las variables que deben ser enteras.
                       Si es None, TODAS son enteras.
    Usa relajación LP + truncado exacto via HiGHS (integrality param).
    """
    n = len(funcion_objetivo)
    c = np.array(funcion_objetivo, dtype=float)
    if tipo_opt == "max":
        c = -c

    integrality = np.zeros(n)
    if variables_enteras is None:
        integrality[:] = 1
    else:
        for idx in variables_enteras:
            integrality[idx] = 1

    res = linprog(c,
                  A_ub=np.array(restricciones, dtype=float),
                  b_ub=np.array(limites, dtype=float),
                  bounds=[(0, None)] * n,
                  integrality=integrality,
                  method="highs")

    if not res.success:
        return {"Estado": "Sin solución", "Mensaje": res.message}

    z = -res.fun if tipo_opt == "max" else res.fun
    variables_dict = {f"X{i+1}": round(v, 4) for i, v in enumerate(res.x)}

    return {"Estado": "Óptimo (Entera)", "Valor_Optimo": round(z, 4),
            "Variables": variables_dict, "Mensaje": res.message}


# ─────────────────────────────────────────────────────────────
# 4. ANÁLISIS DE SENSIBILIDAD (rangos de la FO y del RHS)
# ─────────────────────────────────────────────────────────────
def analizar_sensibilidad(funcion_objetivo, restricciones, limites, tipo_opt="max"):
    """
    Devuelve la solución óptima más un análisis de sensibilidad
    básico: rango en el que cada coeficiente de la FO puede variar
    sin cambiar la base óptima, y rango del RHS de cada restricción.

    Retorna un dict con claves:
        - Solucion_Base   : {'Estado', 'Valor_Optimo', 'Variables'}
        - Sensibilidad_FO : DataFrame con columnas [Variable, c_j, Delta_min, Delta_max]
        - Sensibilidad_RHS: DataFrame con columnas [Restricción, b_i, Delta_min, Delta_max]
        - Holguras        : list de holguras de cada restricción
    """
    base = resolver_programacion_lineal_simplex(funcion_objetivo, restricciones,
                                                limites, tipo_opt)
    if base["Estado"] != "Óptimo":
        return {"Solucion_Base": base}

    n  = len(funcion_objetivo)
    m  = len(limites)
    c0 = np.array(funcion_objetivo, dtype=float)
    A  = np.array(restricciones, dtype=float)
    b  = np.array(limites, dtype=float)
    x0 = np.array([base["Variables"][f"X{i+1}"] for i in range(n)])

    # ── Sensibilidad FO: perturbamos δ en c_j ──────────────────────────────
    delta_step = 0.01
    fo_rows = []
    for j in range(n):
        lo, hi = -1000.0, 1000.0
        for sign, label in [(-1, "lo"), (1, "hi")]:
            d = delta_step
            while abs(d) < 1001:
                c_new = c0.copy(); c_new[j] += sign * d
                r = resolver_programacion_lineal_simplex(c_new.tolist(), A.tolist(),
                                                         b.tolist(), tipo_opt)
                if r["Estado"] != "Óptimo":
                    break
                x_new = np.array([r["Variables"][f"X{i+1}"] for i in range(n)])
                # ¿cambia la base? (alguna variable activa/inactiva cambia)
                base_orig = set(i for i in range(n) if x0[i] > 1e-6)
                base_new  = set(i for i in range(n) if x_new[i] > 1e-6)
                if base_orig != base_new:
                    break
                if sign == -1:
                    lo = -d
                else:
                    hi = d
                d *= 2
        fo_rows.append({"Variable": f"X{j+1}", "c_j": round(c0[j], 4),
                         "Δ_mín": round(lo, 4), "Δ_máx": round(hi, 4)})

    # ── Sensibilidad RHS ────────────────────────────────────────────────────
    rhs_rows = []
    holguras = []
    for i in range(m):
        lhs_val = sum(A[i, j] * x0[j] for j in range(n))
        holgura = round(b[i] - lhs_val, 6)
        holguras.append(holgura)
        lo, hi = -1000.0, 1000.0
        for sign in [-1, 1]:
            d = delta_step
            while abs(d) < 1001:
                b_new = b.copy(); b_new[i] += sign * d
                if any(bv < 0 for bv in b_new):
                    break
                r = resolver_programacion_lineal_simplex(c0.tolist(), A.tolist(),
                                                         b_new.tolist(), tipo_opt)
                if r["Estado"] != "Óptimo":
                    break
                if sign == -1:
                    lo = -d
                else:
                    hi = d
                d *= 2
        rhs_rows.append({"Restricción": f"R{i+1}", "b_i": round(b[i], 4),
                          "Δ_mín": round(lo, 4), "Δ_máx": round(hi, 4)})

    return {
        "Solucion_Base":    base,
        "Sensibilidad_FO":  pd.DataFrame(fo_rows),
        "Sensibilidad_RHS": pd.DataFrame(rhs_rows),
        "Holguras":         holguras,
    }


# ─────────────────────────────────────────────────────────
# 5. PROBLEMA DE TRANSPORTE (Método de la Esquina NO-Oeste)
# ─────────────────────────────────────────────────────────
def resolver_transporte(oferta, demanda, costos):
    """
    Resuelve el Problema de Transporte balanceado.
    oferta  : list  [o1, o2, ..., om]
    demanda : list  [d1, d2, ..., dn]
    costos  : matriz m×n  [[c11,c12,...], ...]

    Utiliza la formulación LP estándar para obtener la solución exacta.
    """
    m = len(oferta)
    n = len(demanda)
    C = np.array(costos, dtype=float).flatten()

    # Restricciones de oferta: sum_j x_ij = o_i
    A_eq = np.zeros((m + n, m * n))
    b_eq = np.array(oferta + demanda, dtype=float)

    for i in range(m):
        for j in range(n):
            A_eq[i, i * n + j] = 1          # oferta

    for j in range(n):
        for i in range(m):
            A_eq[m + j, i * n + j] = 1      # demanda

    res = linprog(C, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * (m * n), method="highs")

    if not res.success:
        return {"Estado": "Sin solución", "Mensaje": res.message}

    X = res.x.reshape(m, n)
    costo_total = round(res.fun, 4)

    tabla = pd.DataFrame(np.round(X, 4),
                         index=[f"Origen {i+1}" for i in range(m)],
                         columns=[f"Destino {j+1}" for j in range(n)])

    return {"Estado": "Óptimo", "Costo_Total": costo_total,
            "Tabla_Asignacion": tabla}


# ─────────────────────────────────────────────────────────
# 6. PROBLEMA DE ASIGNACIÓN (Algoritmo Húngaro / LP)
# ─────────────────────────────────────────────────────────
def resolver_asignacion(costos, tipo_opt="min"):
    """
    Resuelve el Problema de Asignación cuadrado n×n.
    costos  : matriz n×n de costos/beneficios
    tipo_opt: 'min' o 'max'
    """
    C = np.array(costos, dtype=float)
    n = C.shape[0]
    c_flat = C.flatten()
    if tipo_opt == "max":
        c_flat = -c_flat

    A_eq = np.zeros((2 * n, n * n))
    b_eq = np.ones(2 * n)

    for i in range(n):
        for j in range(n):
            A_eq[i,     i * n + j] = 1   # cada agente hace 1 tarea
            A_eq[n + j, i * n + j] = 1   # cada tarea asignada a 1 agente

    res = linprog(c_flat, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, 1)] * (n * n), method="highs")

    if not res.success:
        return {"Estado": "Sin solución", "Mensaje": res.message}

    X = np.round(res.x.reshape(n, n)).astype(int)
    valor = round(-res.fun if tipo_opt == "max" else res.fun, 4)

    asignaciones = [(f"Agente {i+1}", f"Tarea {j+1}", C[i, j])
                    for i in range(n) for j in range(n) if X[i, j] == 1]

    tabla = pd.DataFrame(X,
                         index=[f"Agente {i+1}" for i in range(n)],
                         columns=[f"Tarea {j+1}" for j in range(n)])

    return {"Estado": "Óptimo", "Valor_Optimo": valor,
            "Asignaciones": asignaciones, "Tabla": tabla}