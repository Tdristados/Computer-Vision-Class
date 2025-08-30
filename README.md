# Taller #1 de Visión por computadora — **cvtools**

Librería pequeña de Python con un script de demostración (`main.py`) y pruebas unitarias con `pytest` para ejecutar algoritmos de Computer vision sencillos.

---

## Instalación (Windows, PowerShell)

```powershell
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>

# Dependencias
pip install -r requirements.txt

# Pruebas (opcional)
python -m pytest -q

# Main 
python main.py
```
  
---

## Uso rápido

`main.py` procesa una imagen o un conjunto desde la carpeta `data/`, y puede **mostrar ventanas** (GUI) o **guardar figuras** en archivos. (Idealmente puedes subir tus imagenes a `data/` para poder luego usarlas).

### Parámetros de `main.py`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `--img <ruta>` | string | Procesa una **imagen específica**. |
| `--all` | flag | Procesa **todas** las imágenes ubicadas en `data/`. |
| `--save <carpeta>` | string | **Guarda** todas las figuras en PNG dentro de la carpeta indicada en lugar de abrir ventanas. |

> Recomendación: si su entorno no abre ventanas o desea conservar resultados, use `--save`.

### Ejemplos

```powershell
# 1) Usar la primera imagen encontrada en data/
python main.py

# 2) Procesar una imagen específica
python main.py --img data/ejemplo.png

# 3) Procesar todas las imágenes de data/
python main.py --all

# 4) Guardar resultados en archivos (sin GUI)
python main.py --img data/ejemplo.png --save outputs

# 5) Lote + exportación
python main.py --all --save outputs
```

