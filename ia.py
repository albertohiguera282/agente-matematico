import google.generativeai as genai

def responder(pregunta, api_key):

    # Configurar API
    genai.configure(api_key=api_key)

    # Modelo Gemini
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Prompt
    prompt = f"""
Eres un profesor de matemáticas universitario.
Explica paso a paso:

{pregunta}
"""

    # Generar respuesta
    response = model.generate_content(prompt)

    return response.text