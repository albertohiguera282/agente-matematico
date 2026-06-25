# Script que reemplaza el bloque de optimización en app.py
with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Líneas 791-834 (índices 790-833 base-0) → reemplazar por 3 líneas
new_block = [
    "# ----------------- 7. OPTIMIZACIÓN (P. LINEAL) -----------------\n",
    "elif rama_seleccionada == \"\u2699\ufe0f Optimización (P. Lineal)\":\n",
    "    from optimizacion_app import mostrar_optimizacion\n",
    "    mostrar_optimizacion()\n",
]

patched = lines[:790] + new_block + lines[834:]

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(patched)

print(f"✅ Listo. Líneas originales: {len(lines)} → Líneas nuevas: {len(patched)}")
print("Bloque reemplazado: líneas 791–834 → 4 líneas")
