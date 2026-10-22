import unicodedata
import re
import pandas as pd
import numpy as np
from scipy import stats


def higea_remove_accents(text):
    """Remueve tildes y caracteres diacríticos de una cadena de texto."""
    if not isinstance(text, str):
        return text
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def higea_clean_column_names(df):
    """Estandariza los nombres de columnas a minusculas, sin tildes ni caracteres especiales."""
    if df is None:
        return None
    out = df.copy(deep=True)
    new_cols = []
    for col in out.columns:
        c_clean = str(col).strip()
        c_clean = higea_remove_accents(c_clean).lower()
        c_clean = re.sub(r'[^a-z0-9_]+', '_', c_clean)
        c_clean = re.sub(r'_+', '_', c_clean).strip('_')
        new_cols.append(c_clean if c_clean else f"col_{len(new_cols)}")
    out.columns = new_cols
    return out


def higea_normalize_text_case(df, cols, mode="lower", remove_accents=False):
    """Normaliza el formato de texto en las columnas seleccionadas."""
    if df is None or not cols:
        return df
    out = df.copy(deep=True)
    for col in cols:
        if col in out.columns:
            s = out[col].astype(str)
            if remove_accents:
                s = s.map(higea_remove_accents)
            if mode == "lower":
                out[col] = s.str.lower()
            elif mode == "upper":
                out[col] = s.str.upper()
            elif mode == "title":
                out[col] = s.str.title()
    return out


def higea_convert_column_type(df, col, target_type):
    """Convierte el tipo de datos de una columna (numeric, category, string, datetime)."""
    if df is None or col not in df.columns:
        return df
    out = df.copy(deep=True)
    try:
        if target_type == "numeric":
            out[col] = pd.to_numeric(out[col], errors='coerce')
        elif target_type == "category":
            out[col] = out[col].astype("category")
        elif target_type == "string":
            out[col] = out[col].astype(str)
        elif target_type == "datetime":
            out[col] = pd.to_datetime(out[col], errors='coerce')
    except Exception:
        pass
    return out


def higea_scale_column(df, col, method="zscore"):
    """Escala o estandariza una columna numérica usando Z-Score o Min-Max."""
    if df is None or col not in df.columns:
        return df
    out = df.copy(deep=True)
    if not pd.api.types.is_numeric_dtype(out[col]):
        return out
    series = out[col]
    if method == "zscore":
        std = series.std()
        if std != 0:
            out[col] = (series - series.mean()) / std
    elif method == "minmax":
        min_val = series.min()
        max_val = series.max()
        if max_val != min_val:
            out[col] = (series - min_val) / (max_val - min_val)
    return out


class AS001DiagnosticAssistant:
    """Asistente de Diagnóstico del Dataset (AS-001) para inspección de calidad e integridad."""

    def __init__(self, df=None):
        self.df = df
        self.report = {}

    def set_dataframe(self, df):
        self.df = df

    def run_full_diagnosis(self):
        """Ejecuta una auditoría completa del conjunto de datos."""
        if self.df is None or self.df.empty:
            return {"error": "Dataset no cargado o vacío"}

        df = self.df
        total_rows = len(df)
        total_cols = len(df.columns)
        total_missing = int(df.isna().sum().sum())
        total_duplicates = int(df.duplicated().sum())

        col_details = []
        for col in df.columns:
            series = df[col]
            missing_col = int(series.isna().sum())
            pct_missing = (missing_col / total_rows) * 100 if total_rows > 0 else 0
            is_num = pd.api.types.is_numeric_dtype(series)

            col_details.append({
                "Columna": col,
                "Tipo": "Numérica" if is_num else "Categórica",
                "Nulos": missing_col,
                "% Nulos": f"{pct_missing:.1f}%",
                "Unicos": int(series.nunique(dropna=True))
            })

        # Puntaje de calidad (0 a 100%)
        missing_penalty = (total_missing / (total_rows * total_cols)) * 50 if (total_rows * total_cols) > 0 else 0
        dup_penalty = (total_duplicates / total_rows) * 50 if total_rows > 0 else 0
        quality_score = max(0.0, min(100.0, 100.0 - missing_penalty - dup_penalty))

        self.report = {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "total_missing": total_missing,
            "total_duplicates": total_duplicates,
            "quality_score": round(quality_score, 1),
            "column_details": pd.DataFrame(col_details)
        }
        return self.report

    def detect_outliers(self, col_name, method="iqr"):
        """Detecta valores atípicos en una columna numérica según IQR o Z-score."""
        if self.df is None or col_name not in self.df.columns:
            return pd.DataFrame()
        
        series = self.df[col_name].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            return pd.DataFrame()

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_mask = (series < lower_bound) | (series > upper_bound)
        else: # zscore
            z_scores = np.abs(stats.zscore(series))
            outliers_mask = z_scores > 3.0

        outliers_df = self.df.loc[series[outliers_mask].index].copy()
        return outliers_df


def higea_auto_clean(df):
    """Ejecuta una limpieza integral automática de 1-clic:
    1. Quita espacios en nombres y valores.
    2. Elimina filas/columnas 100% vacías.
    3. Elimina copias duplicadas.
    4. Imputa faltantes numéricos con la mediana y categóricos con la moda.
    """
    if df is None:
        return None
    
    out = df.copy(deep=True)
    # 1. Espacios
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        
    # 2. Filas y columnas vacías
    out = out.dropna(how="all").dropna(axis=1, how="all")
    
    # 3. Duplicados
    out = out.drop_duplicates(keep="first")
    
    # 4. Imputación
    for col in out.columns:
        if out[col].isna().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                mediana = out[col].median()
                if pd.notna(mediana):
                    out[col] = out[col].fillna(mediana)
            else:
                moda = out[col].mode(dropna=True)
                if not moda.empty:
                    out[col] = out[col].fillna(moda.iloc[0])
                    
    return out.copy()


def higea_winsorize_column(series, limits=(0.05, 0.05)):
    """Ajusta valores extremos al percentil especificado (Winsorización)."""
    if not pd.api.types.is_numeric_dtype(series):
        return series
    s_clean = series.dropna()
    low_val = s_clean.quantile(limits[0])
    high_val = s_clean.quantile(1.0 - limits[1])
    return series.clip(lower=low_val, upper=high_val)


class AS002MethodologicalAssistant:
    """Asistente Metodológico (AS-002) para guiar el diseño experimental."""

    def __init__(self, df=None):
        self.df = df

    def suggest_research_design(self, num_vars, target_type):
        if target_type == "numeric":
            return "Diseño Cuantitativo Observacional o Experimental con Análisis Varianza / Regresión"
        else:
            return "Diseño Categórico / Tablas de Contingencia y Modelos Logísticos"


def initialize_higea_context():
    return {"status": "HIGEA Core Engine Initialized"}


