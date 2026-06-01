import statistics

def descriptiva(datos):

    resultado = {}

    # Cantidad de datos
    resultado["Cantidad"] = len(datos)

    # Tendencia central
    resultado["Media"] = statistics.mean(datos)
    resultado["Mediana"] = statistics.median(datos)

    try:
        resultado["Moda"] = statistics.mode(datos)
    except:
        resultado["Moda"] = "No existe una moda única"

    # Dispersión
    resultado["Minimo"] = min(datos)
    resultado["Maximo"] = max(datos)
    resultado["Rango"] = max(datos) - min(datos)

    if len(datos) >= 2:
        resultado["Varianza"] = statistics.variance(datos)
        resultado["Desviacion_Estandar"] = statistics.stdev(datos)

        media = statistics.mean(datos)

        if media != 0:
            resultado["Coeficiente_Variacion (%)"] = round(
                (statistics.stdev(datos) / media) * 100, 2
            )
        else:
            resultado["Coeficiente_Variacion (%)"] = "No aplica"

    else:
        resultado["Varianza"] = "Se requieren al menos 2 datos"
        resultado["Desviacion_Estandar"] = "Se requieren al menos 2 datos"
        resultado["Coeficiente_Variacion (%)"] = "No aplica"

    return resultado

