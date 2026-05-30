import streamlit as st
from ia import responder
import sympy as sp
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Obtener la clave de API de las variables de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ====================================
# CONFIGURACIÓN GENERAL
# ====================================
st.set_page_config(
    page_title="Agente Matemático",
    layout="wide"
)

# ====================================
# MENÚ LATERAL
# ====================================
opcion = st.sidebar.selectbox(
    "Seleccione un módulo",
    [
        "🏠 Inicio",
        "🧠 IA Matemática"
    ]
)

# ====================================
# PANTALLA INICIO
# ====================================
if opcion == "🏠 Inicio":

    st.title("🧠 Agente Matemático Universitario")
    st.write("Sistema funcionando correctamente 🚀")

# ====================================
# IA MATEMÁTICA
# ====================================
elif opcion == "🧠 IA Matemática":

    st.title("Profesor Matemático IA")

    pregunta = st.text_area(
        "Escribe tu pregunta matemática"
    )

    if st.button("Resolver con IA"):

        if pregunta:

            try:

                respuesta = responder(
                    pregunta,
                    GEMINI_API_KEY
                )

                st.success(respuesta)

            except Exception as e:

                st.error(f"Error: {e}")

        else:

            st.warning(
                "Escribe una pregunta antes de continuar"
            )
            

x = sp.symbols('x')

ecuacion = sp.solve(2*x + 10, x)

st.write(ecuacion)
st.write("Resultado de la ecuación")

st.success(ecuacion)

st.subheader("Más ejercicios")

