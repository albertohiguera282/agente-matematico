# Contenido definitivo y único de finanzas.py

def amortizacion_cuota_fija(capital, tasa_nominal, meses_plazo):
    """Cálculo de amortización con cuotas mensuales fijas (Sistema de Cuota Fija)."""
    tasa_mensual = (tasa_nominal / 100) / 12
    cuota = capital * (tasa_mensual * (1 + tasa_mensual)**meses_plazo) / (((1 + tasa_mensual)**meses_plazo) - 1)
    
    tabla = []
    saldo = capital
    for mes in range(1, meses_plazo + 1):
        interes = saldo * tasa_mensual
        amortizacion_capital = cuota - interes
        saldo -= amortizacion_capital
        tabla.append({
            "Mes": mes,
            "Cuota": round(cuota, 2),
            "Interés": round(interes, 2),
            "Amortización": round(amortizacion_capital, 2),
            "Saldo_Restante": round(max(0, saldo), 2)
        })
    return tabla, cuota


def amortizacion_cuota_decreciente(capital, tasa_nominal, meses_plazo):
    """Cálculo de amortización con abono constante a capital (Sistema de Amortización Constante)."""
    tasa_mensual = (tasa_nominal / 100) / 12
    amortizacion_capital = capital / meses_plazo
    
    tabla = []
    saldo = capital
    for mes in range(1, meses_plazo + 1):
        interes = saldo * tasa_mensual
        cuota = amortizacion_capital + interes
        saldo -= amortizacion_capital
        tabla.append({
            "Mes": mes,
            "Cuota": round(cuota, 2),
            "Interés": round(interes, 2),
            "Amortización": round(amortizacion_capital, 2),
            "Saldo_Restante": round(max(0, saldo), 2)
        })
    return tabla


def evaluar_proyecto_van_tir(inversion_inicial, flujos, tasa_descuento):
    """
    Calcula el VAN (Valor Actual Neto) y el TIR (Tasa Interna de Retorno)
    de un proyecto a partir de su inversión inicial y sus flujos de caja futuros.

    - inversion_inicial: monto invertido en el periodo 0 (número positivo).
    - flujos: lista de flujos de caja netos para los periodos 1, 2, 3, ...
    - tasa_descuento: tasa de descuento anual en porcentaje (ej. 10 para 10%).
    """
    tasa = tasa_descuento / 100

    # Valor Actual Neto
    van = -inversion_inicial
    detalle_flujos = []
    for periodo, flujo in enumerate(flujos, start=1):
        valor_presente = flujo / ((1 + tasa) ** periodo)
        van += valor_presente
        detalle_flujos.append({
            "Periodo": periodo,
            "Flujo_Neto": round(flujo, 2),
            "Valor_Presente": round(valor_presente, 2)
        })

    # Tasa Interna de Retorno (búsqueda numérica con bisección)
    def van_con_tasa(r):
        valor = -inversion_inicial
        for periodo, flujo in enumerate(flujos, start=1):
            valor += flujo / ((1 + r) ** periodo)
        return valor

    tir = None
    r_bajo, r_alto = -0.99, 10.0
    van_bajo, van_alto = van_con_tasa(r_bajo), van_con_tasa(r_alto)

    if van_bajo * van_alto < 0:
        for _ in range(200):
            r_medio = (r_bajo + r_alto) / 2
            van_medio = van_con_tasa(r_medio)
            if abs(van_medio) < 1e-6:
                break
            if van_bajo * van_medio < 0:
                r_alto = r_medio
                van_alto = van_medio
            else:
                r_bajo = r_medio
                van_bajo = van_medio
        tir = r_medio * 100

    if van > 0:
        decision = "Proyecto VIABLE: el VAN es positivo, se recomienda aceptar el proyecto."
    elif van < 0:
        decision = "Proyecto NO VIABLE: el VAN es negativo, se recomienda rechazar el proyecto."
    else:
        decision = "Proyecto INDIFERENTE: el VAN es exactamente cero."

    return {
        "VAN": round(van, 2),
        "TIR": round(tir, 4) if tir is not None else None,
        "Detalle_Flujos": detalle_flujos,
        "Decision": decision
    }


def amortizacion_gradiente_aritmetico(capital, tasa_nominal, meses_plazo, gradiente_valor, es_creciente=True):
    tasa_mensual = (tasa_nominal / 100) / 12
    signo = 1 if es_creciente else -1
    
    if tasa_mensual > 0:
        S1 = sum((1 + tasa_mensual) ** (-t) for t in range(1, meses_plazo + 1))
        S2 = sum((t - 1) * (1 + tasa_mensual) ** (-t) for t in range(1, meses_plazo + 1))
    else:
        S1 = meses_plazo
        S2 = meses_plazo * (meses_plazo - 1) / 2
        
    a1 = (capital - signo * gradiente_valor * S2) / S1
    
    tabla = []
    saldo = capital
    for mes in range(1, meses_plazo + 1):
        cuota = a1 + signo * (mes - 1) * gradiente_valor
        interes = saldo * tasa_mensual
        amortizacion_capital = cuota - interes
        saldo -= amortizacion_capital
        tabla.append({
            "Mes": mes,
            "Cuota": round(cuota, 2),
            "Interés": round(interes, 2),
            "Amortización": round(amortizacion_capital, 2),
            "Saldo_Restante": round(max(0.0, saldo), 2)
        })
    return tabla, a1


def amortizacion_gradiente_geometrico(capital, tasa_nominal, meses_plazo, gradiente_porcentaje, es_creciente=True):
    tasa_mensual = (tasa_nominal / 100) / 12
    g = gradiente_porcentaje / 100
    signo_g = 1 + g if es_creciente else 1 - g
    
    S = sum((signo_g ** (t - 1)) / ((1 + tasa_mensual) ** t) for t in range(1, meses_plazo + 1))
    a1 = capital / S
    
    tabla = []
    saldo = capital
    for mes in range(1, meses_plazo + 1):
        cuota = a1 * (signo_g ** (mes - 1))
        interes = saldo * tasa_mensual
        amortizacion_capital = cuota - interes
        saldo -= amortizacion_capital
        tabla.append({
            "Mes": mes,
            "Cuota": round(cuota, 2),
            "Interés": round(interes, 2),
            "Amortización": round(amortizacion_capital, 2),
            "Saldo_Restante": round(max(0.0, saldo), 2)
        })
    return tabla, a1


def evaluar_proyecto(inversion_inicial, flujos, tasa_descuento):
    res = evaluar_proyecto_van_tir(inversion_inicial, flujos, tasa_descuento)
    return res["VAN"], res["TIR"], res["Detalle_Flujos"]

