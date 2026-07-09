# Agente Estadístico — Consultor Inteligente de Modelos Estadísticos

Aplicación web académica/científica construida con **Streamlit** para diagnóstico, selección y ejecución de análisis estadísticos. Diseñada con rigor metodológico de nivel doctoral.

---

## 🎯 ¿Qué hace esta aplicación?

La aplicación opera en **4 módulos integrados** accesibles desde la barra lateral:

| Módulo | Propósito |
|--------|-----------|
| **Asesor Metodológico** | Wizard de 5 pasos que diagnostica qué modelo estadístico usar según tu objetivo, tipo de variables, diseño muestral y supuestos |
| **Biblioteca de Modelos** | Catálogo navegable de **300+ modelos** (12 módulos temáticos) con fórmulas LaTeX, supuestos, justificaciones y plantillas de código en **Python, R, SPSS, Stata, SAS** |
| **Laboratorio de Datos** | Subida de CSV → detección automática de tipos de variable → estadísticos descriptivos + visualizaciones Plotly (histogramas, boxplots, scatterplots, heatmaps) |
| **Calculadora Analítica** | 6 pruebas estadísticas con ingreso manual o desde datos: descriptivos, t-Student (independientes/pareados), chi-cuadrado, regresión lineal simple, ANOVA una vía — resultados con interpretación científica y generación de código reproducible |

---

## 🧮 Pruebas Estadísticas Implementadas

| Prueba | Función | Librería base |
|--------|---------|---------------|
| Estadísticos descriptivos (media, DE, IC 95%, skewness, kurtosis, IQR) | `calculate_descriptive()` | numpy + scipy |
| t de Student (muestras independientes + Welch) | `calculate_independent_ttest()` | Propia + scipy.stats |
| t pareado (por diferencias) | `calculate_paired_ttest()` | Propia + scipy.stats |
| Chi-cuadrado de independencia | `calculate_chi_squared()` | scipy.stats.chi2_contingency |
| Regresión lineal simple + ANOVA de la regresión | `calculate_simple_regression()` | scipy.stats.linregress |
| ANOVA una vía | `calculate_oneway_anova()` | Cálculo propio (SS between/within) |

Todas las calculadoras devuelven **fórmulas LaTeX paso a paso** y **interpretación científica automática**.

---

## 📚 Catálogo de Modelos (300+)

Generado programáticamente en `database.py`:

- **12 módulos temáticos**: Descriptiva, Probabilidad, Inferencia, Regresión, Correlación, Multivariado, Series de Tiempo, ANOVA, Supervivencia, SEM, Econometría Avanzada, Machine Learning
- **7 modelos "Core"** con detalle completo (justificación, supuestos, LaTeX, código ×5 lenguajes)
- **42 modelos base** adicionales
- **7 variaciones metodológicas**: Frecuentista, Welch, Bootstrap, Bayesiano, FDR, Bonferroni, Huber
- **Total: ~301 modelos** (42 × 7 + 7 core)

---

## 🛠 Stack Tecnológico

| Capa | Tecnología |
|------|------------|
| Lenguaje | Python 3.13 |
| UI Framework | Streamlit |
| Datos | pandas, numpy |
| Estadística | scipy.stats |
| Visualización | plotly (express + graph_objects) |
| Matemáticas | LaTeX via `st.latex` |
| Estilos | CSS custom (dark mode premium, glassmorphism) |

---

## 🚀 Instalación y Ejecución

```bash
# 1. Clonar repositorio
git clone <url-del-repo>
cd agente_estadistico

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar aplicación
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

---

## 📦 Dependencias

Ver `requirements.txt` para la lista exacta. Principales:

```
streamlit>=1.35
pandas>=2.2
numpy>=1.26
scipy>=1.13
plotly>=5.22
```

---

## ✅ Validación Matemática

El archivo `tests.py` incluye **6 pruebas unitarias** que comparan los resultados del calculador contra valores de referencia conocidos:

1. Descriptivos (media, DE, varianza)
2. t de muestras independientes (t, p-valor)
3. t pareado (t, p-valor)
4. Regresión lineal (pendiente, intercepto, R², F)
5. ANOVA una vía (F, p-valor)
6. Chi-cuadrado (consistencia básica)

Al iniciar la app, se ejecuta la validación y muestra un badge **"Rigor Matemático Validado (100% PASS)"** en la barra lateral.

---

## 📁 Estructura del Proyecto

```
agente_estadistico/
├── app.py              # Aplicación principal Streamlit (1033 líneas)
├── advisor.py          # Árbol de decisión metodológico (186 líneas)
├── calculator.py       # Motor de cálculos + LaTeX (440 líneas)
├── data_processor.py   # Parser CSV + detector de tipos (98 líneas)
├── database.py         # Catálogo 300+ modelos (471 líneas)
├── tests.py            # Validación matemática (173 líneas)
├── requirements.txt    # Dependencias Python
├── .gitignore          # Archivos ignorados por Git
└── README.md           # Este archivo
```

---

## 🎓 Uso Académico / Científico

- **Nivel**: Doctoral / Investigación avanzada
- **Enfoque**: Reproducibilidad, transparencia metodológica, rigor matemático
- **Output**: Código ejecutable en 5 lenguajes + fórmulas LaTeX + interpretación en lenguaje natural
- **Sin IA generativa**: El asesor es un árbol de decisión determinístico codificado (auditable, explicable)

---

## 📄 Licencia

[Por definir — sugerencia: MIT o Apache 2.0]

---

## 🤝 Contribuciones

1. Fork del repositorio
2. Crear rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit (`git commit -m 'feat: descripción del cambio'`)
4. Push (`git push origin feature/nueva-funcionalidad`)
5. Abrir Pull Request

---

## 📞 Contacto / Autor

**Alberto** — Proyecto personal de consultoría estadística