import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ─────────────────────────────────────────────
# 1. DISTANCIA Y RECTA ENTRE DOS PUNTOS
# ─────────────────────────────────────────────
def distancia_y_recta(x1, y1, x2, y2):
    distancia = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if x2 - x1 == 0:
        pendiente = None
        ecuacion = f"x = {x1}"
    else:
        pendiente = (y2 - y1) / (x2 - x1)
        intercepto = y1 - pendiente * x1
        ecuacion = f"y = {pendiente:.4f}x + {intercepto:.4f}"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([x1, x2], [y1, y2], color="#00BFFF", lw=2, marker="o", markersize=8)
    ax.text(x1, y1, f"  A({x1},{y1})", color="white", fontsize=10)
    ax.text(x2, y2, f"  B({x2},{y2})", color="white", fontsize=10)
    ax.set_title("Distancia y Recta entre dos Puntos", color="white")
    ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white")
    ax.grid(color="#333333", linestyle=":", alpha=0.5)
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#1E1E24")
    for s in ax.spines.values(): s.set_color("#333333")

    return {"Distancia": round(distancia, 6), "Pendiente": pendiente,
            "Ecuacion": ecuacion, "Fig": fig}

# ─────────────────────────────────────────────
# 2. CÍRCULO UNITARIO
# ─────────────────────────────────────────────
def graficar_circulo_unitario(angulo_grados):
    angulo_rad = np.radians(angulo_grados)
    cos_val = np.cos(angulo_rad)
    sin_val = np.sin(angulo_rad)

    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#9B5DE5", lw=2)
    ax.axhline(0, color="white", lw=0.8); ax.axvline(0, color="white", lw=0.8)
    ax.plot([0, cos_val], [0, sin_val], color="#F15BB5", lw=2)
    ax.scatter([cos_val], [sin_val], color="#00F5D4", s=80, zorder=5)
    ax.text(cos_val + 0.05, sin_val + 0.05,
            f"({cos_val:.3f}, {sin_val:.3f})", color="white", fontsize=9)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title(f"Círculo Unitario — {angulo_grados}°", color="white")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#1E1E24")
    for s in ax.spines.values(): s.set_color("#333333")

    return {"Angulo_rad": round(angulo_rad, 6),
            "cos": round(cos_val, 6), "sin": round(sin_val, 6),
            "tan": round(np.tan(angulo_rad), 6) if cos_val != 0 else None,
            "Fig": fig}

# ─────────────────────────────────────────────
# 3. SIMPLIFICAR IDENTIDAD TRIGONOMÉTRICA
# ─────────────────────────────────────────────
def simplificar_identidad_trig(expresion_str):
    x = sp.Symbol("x", real=True)
    expr = sp.sympify(expresion_str, locals={"x": x,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sec": sp.sec, "csc": sp.csc, "cot": sp.cot})
    simplificada = sp.trigsimp(expr)
    return {"Original": str(expr), "Simplificada": str(simplificada),
            "LaTeX": sp.latex(simplificada)}

# ─────────────────────────────────────────────
# 4. RESOLVER ECUACIÓN TRIGONOMÉTRICA
# ─────────────────────────────────────────────
def resolver_ecuacion_trig(ecuacion_str, intervalo=(0, 2*np.pi)):
    x = sp.Symbol("x", real=True)
    try:
        if "=" in ecuacion_str:
            lhs, rhs = ecuacion_str.split("=")
            expr = sp.sympify(lhs.strip(), locals={"x": x,
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan}) - \
                   sp.sympify(rhs.strip(), locals={"x": x,
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan})
        else:
            expr = sp.sympify(ecuacion_str, locals={"x": x,
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan})

        soluciones_gen = sp.solve(expr, x)
        a, b = intervalo
        soluciones_intervalo = []
        for sol in soluciones_gen:
            val = float(sol.evalf())
            while val < a:
                val += 2 * np.pi
            while val <= b + 1e-9:
                if val >= a - 1e-9:
                    soluciones_intervalo.append(round(val, 6))
                val += 2 * np.pi

        return {"Soluciones_generales": [str(s) for s in soluciones_gen],
                "Soluciones_intervalo": sorted(set(soluciones_intervalo)),
                "Intervalo": f"[{a:.4f}, {b:.4f}]"}
    except Exception as e:
        return {"Error": str(e)}

# ─────────────────────────────────────────────
# 5. GRAFICAR ONDA TRIGONOMÉTRICA
# ─────────────────────────────────────────────
def graficar_onda_trig(amplitud, frecuencia, fase, funcion="sin"):
    x = np.linspace(0, 4 * np.pi, 500)
    if funcion == "sin":
        y = amplitud * np.sin(frecuencia * x + fase)
    elif funcion == "cos":
        y = amplitud * np.cos(frecuencia * x + fase)
    else:
        y = amplitud * np.tan(frecuencia * x + fase)
        y = np.where(np.abs(y) > 10, np.nan, y)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="#00F5D4", lw=2)
    ax.axhline(0, color="white", lw=0.8)
    ax.set_title(f"y = {amplitud}·{funcion}({frecuencia}x + {fase})", color="white")
    ax.set_xlabel("x (rad)", color="white"); ax.set_ylabel("y", color="white")
    ax.grid(color="#333333", linestyle=":", alpha=0.5)
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#1E1E24")
    for s in ax.spines.values(): s.set_color("#333333")

    periodo = round(2 * np.pi / frecuencia, 4) if frecuencia != 0 else None
    return {"Amplitud": amplitud, "Frecuencia": frecuencia,
            "Fase": fase, "Periodo": periodo, "Fig": fig}

# ─────────────────────────────────────────────
# 6. RESOLVER TRIÁNGULO (Ley de Senos/Cosenos)
# ─────────────────────────────────────────────
def resolver_triangulo(a=None, b=None, c=None,
                        A=None, B=None, C=None):
    """
    Recibe al menos 3 datos (lados y/o ángulos en grados).
    Devuelve los 6 elementos del triángulo y un gráfico.
    """
    # Convertir ángulos a radianes
    Ar = np.radians(A) if A is not None else None
    Br = np.radians(B) if B is not None else None
    Cr = np.radians(C) if C is not None else None

    # Completar ángulo faltante si se tienen dos
    if Ar is not None and Br is not None and Cr is None:
        Cr = np.pi - Ar - Br
        C = np.degrees(Cr)
    elif Ar is not None and Cr is not None and Br is None:
        Br = np.pi - Ar - Cr
        B = np.degrees(Br)
    elif Br is not None and Cr is not None and Ar is None:
        Ar = np.pi - Br - Cr
        A = np.degrees(Ar)

    # Ley de Senos para completar lados
    if a is not None and Ar is not None:
        k = a / np.sin(Ar)
        if b is None and Br is not None: b = k * np.sin(Br)
        if c is None and Cr is not None: c = k * np.sin(Cr)
    elif b is not None and Br is not None:
        k = b / np.sin(Br)
        if a is None and Ar is not None: a = k * np.sin(Ar)
        if c is None and Cr is not None: c = k * np.sin(Cr)
    elif c is not None and Cr is not None:
        k = c / np.sin(Cr)
        if a is None and Ar is not None: a = k * np.sin(Ar)
        if b is None and Br is not None: b = k * np.sin(Br)

    # Ley de Cosenos si faltan ángulos
    if a is not None and b is not None and c is not None:
        if Ar is None:
            Ar = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
            A = np.degrees(Ar)
        if Br is None:
            Br = np.arccos((a**2 + c**2 - b**2) / (2*a*c))
            B = np.degrees(Br)
        if Cr is None:
            Cr = np.pi - Ar - Br
            C = np.degrees(Cr)

    area = 0.5 * a * b * np.sin(Cr) if (a and b and Cr) else None

    # Gráfico
    fig, ax = plt.subplots(figsize=(6, 5))
    if a and b and Cr:
        P1 = np.array([0, 0])
        P2 = np.array([c, 0])
        P3 = np.array([b * np.cos(Ar), b * np.sin(Ar)])
        triangulo = plt.Polygon([P1, P2, P3],
                                 fill=True, facecolor="#9B5DE520",
                                 edgecolor="#9B5DE5", lw=2)
        ax.add_patch(triangulo)
        for P, label in zip([P1, P2, P3], ["A", "B", "C"]):
            ax.scatter(*P, color="#F15BB5", s=60, zorder=5)
            ax.text(P[0], P[1] + 0.05, f"  {label}", color="white", fontsize=10)
        mid = max(c, b) * 1.2 if c and b else 10
        ax.set_xlim(-mid*0.2, mid*1.1); ax.set_ylim(-mid*0.2, mid*1.1)
    ax.set_aspect("equal")
    ax.set_title("Triángulo Resuelto", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", linestyle=":", alpha=0.4)
    fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#1E1E24")
    for s in ax.spines.values(): s.set_color("#333333")

    return {
        "Lado_a": round(a, 4) if a else None,
        "Lado_b": round(b, 4) if b else None,
        "Lado_c": round(c, 4) if c else None,
        "Angulo_A": round(A, 4) if A else None,
        "Angulo_B": round(B, 4) if B else None,
        "Angulo_C": round(C, 4) if C else None,
        "Area": round(area, 4) if area else None,
        "Fig": fig
    }