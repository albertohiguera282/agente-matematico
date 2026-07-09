# -*- coding: utf-8 -*-
"""
CSV parser and intelligent variable type detector.
"""

import pandas as pd
import numpy as np

def parse_csv(file_obj):
    """
    Parses a CSV file object, automatically detecting if the delimiter is comma or semicolon.
    """
    try:
        # Sniff separator by reading the first line
        first_line = file_obj.readline()
        # Reset file pointer
        file_obj.seek(0)
        
        # Decode first line to string if bytes
        if isinstance(first_line, bytes):
            first_line = first_line.decode('utf-8', errors='ignore')
            
        semicolon_count = first_line.count(';')
        comma_count = first_line.count(',')
        
        sep = ';' if semicolon_count > comma_count else ','
        
        # Read CSV
        df = pd.read_csv(file_obj, sep=sep, skipinitialspace=True)
        
        # Clean column names (strip quotes and spaces)
        df.columns = [col.strip().replace('"', '').replace("'", "") for col in df.columns]
        
        # Clean string cells
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.replace('"', '').str.replace("'", "")
            
        return df
    except Exception as e:
        raise ValueError(f"Error al parsear el archivo CSV: {str(e)}")

def detect_variable_types(df):
    """
    Classifies each column as 'numeric' or 'categorical'.
    A column is 'numeric' if >= 85% of non-null values can be converted to numbers.
    """
    types_dict = {}
    cleaned_df = df.copy()
    
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            types_dict[col] = "categorical"  # Empty columns fallback to categorical
            continue
            
        # Try to convert to numeric
        converted = pd.to_numeric(series, errors='coerce')
        valid_nums = converted.notnull().sum()
        pct_numeric = valid_nums / len(series)
        
        if pct_numeric >= 0.85:
            types_dict[col] = "numeric"
            # Replace column in cleaned_df with numeric values
            cleaned_df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            types_dict[col] = "categorical"
            cleaned_df[col] = df[col].astype(str)
            
    return cleaned_df, types_dict

def get_variable_summary(series, var_type):
    """
    Returns a small quick dictionary of stats for previewing.
    """
    series_clean = series.dropna()
    n = len(series_clean)
    if n == 0:
        return {"n": 0}
        
    if var_type == "numeric":
        return {
            "n": n,
            "mean": float(series_clean.mean()),
            "std": float(series_clean.std(ddof=1)) if n > 1 else 0.0,
            "min": float(series_clean.min()),
            "max": float(series_clean.max()),
            "median": float(series_clean.median())
        }
    else:
        counts = series_clean.value_counts()
        most_freq = counts.index[0] if not counts.empty else "N/A"
        most_freq_pct = (counts.iloc[0] / n * 100) if not counts.empty else 0.0
        return {
            "n": n,
            "unique_categories": len(counts),
            "most_frequent": str(most_freq),
            "most_frequent_pct": float(most_freq_pct)
        }
