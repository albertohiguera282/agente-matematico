import os
import google.generativeai as genai

def cargar_api_key():
    # Intentar cargar desde el entorno
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    # Intentar cargar leyendo manualmente .env si existe
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "GEMINI_API_KEY":
                            return v.strip().strip('"').strip("'")
    return None

def responder(pregunta, api_key=None, tema=None):
    # Obtener API Key
    if not api_key:
        api_key = cargar_api_key()
        
    if not api_key:
        raise ValueError(
            "API Key de Gemini no encontrada. Por favor, configúrala en el archivo .env o en la barra lateral."
        )

    # Configurar API
    genai.configure(api_key=api_key)

    # Modelo Gemini
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Construir el prompt con instrucciones estructuradas para el Agente Matemático
    contexto_tema = f" en el contexto de '{tema}'" if tema else ""
    prompt = rf"""
Eres un Agente Matemático Universitario de Inteligencia Artificial altamente capacitado. 
Tu objetivo es resolver el siguiente ejercicio matemático{contexto_tema} de manera impecable, rigurosa y didáctica.

**Pregunta/Ejercicio:**
{pregunta}

**Instrucciones obligatorias de formato y contenido:**
1. **Razonamiento Paso a Paso (Chain of Thought):** Divide tu explicación en pasos claros y ordenados lógicamente (ej: Paso 1, Paso 2).
2. **Uso de LaTeX:** Escribe TODAS las fórmulas matemáticas y ecuaciones utilizando la sintaxis de LaTeX. 
   - Usa `$$ ... $$` en líneas independientes para ecuaciones grandes o importantes.
   - Usa `$ ... $` dentro de los párrafos para variables, términos y fórmulas cortas en línea (ej: $f(x) = x^2$, $\sigma$, $\theta$, $x_1$).
3. **Claridad Conceptual:** Explica brevemente el fundamento teórico detrás de cada paso o teorema aplicado.
4. **Respuesta Final:** Destaca claramente la respuesta o conclusión final del ejercicio.
5. **Idioma:** Responde completamente en español.

Genera tu respuesta con un formato markdown limpio y profesional.
"""

    # Generar respuesta
    response = model.generate_content(prompt)

    return response.text