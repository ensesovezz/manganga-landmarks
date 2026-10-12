# Agregar datos nuevos (abejas de Uruguay)

Guía para incorporar fotos propias al pipeline y reentrenar el modelo.

## 1. Fotografiar

Seguí el [protocolo de fotografía](../README.md#protocolo-de-fotografía):

- ala anterior **derecha** (el esquema de 19 puntos está definido para esa ala);
- ala plana, sin pliegues;
- luz difusa y uniforme;
- escala constante entre muestras.

## 2. Digitalizar los landmarks

Con [tpsDig2](https://sbmorphometrics.org/) o [MorphoJ](https://www.flywings.org.uk/morphoj_page.htm):

1. Abrí la imagen del ala.
2. Marcá los 19 puntos **en el mismo orden** que define la referencia
   (`manganga/assets/reference_19.txt`). Para saber a qué punto corresponde cada
   índice, mirá `resultados/deteccoes/main-reference.txt` o cualquier overlay generado
   con `manganga detect` sobre una muestra ya conocida.
3. Exportá el archivo `.tps`.

> Si tenés dudas sobre el orden, generá un overlay con el modelo actual y comparalo
> con tu criterio anatómico: los números se dibujan sobre la imagen.

## 3. Generar las máscaras de entrenamiento

El modelo no se entrena con los puntos sino con **discos** en la posición de los puntos
(así funcionaba el pipeline original). Usá `cria_mascara.py` del repo o el script de
conversión:

```bash
python - <<'PY'
import numpy as np, cv2
from pathlib import Path
from manganga.io import read_tps

# Ajustá estas rutas
TPS_DIR = Path("data/raw/tps")
IMG_DIR = Path("data/raw/aladas")   # mismas alas recortadas y normalizadas
OUT_DIR = Path("data/processed/masks")
RADIUS = 4                          # mismo radio que el dataset original

OUT_DIR.mkdir(parents=True, exist_ok=True)
for tps_path in sorted(TPS_DIR.glob("*.tps")):
    points, image_id = read_tps(tps_path)
    img_path = IMG_DIR / (image_id or f"{tps_path.stem}.jpg")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("falta la imagen:", img_path)
        continue
    h, w = img.shape
    mask = np.zeros((h, w), np.uint8)
    for x, y in points:
        cv2.circle(mask, (int(round(x)), int(round(y))), RADIUS, 255, -1)
    cv2.imwrite(str(OUT_DIR / img_path.name), mask)
PY
```

Si las fotos son crudas (no recortadas), usá `manganga.crop.crop_wing` primero para
llevarlas a 400x400, **y marcá los puntos sobre la imagen ya recortada**.

## 4. Validar el nuevo ground truth

```bash
manganga validate-data --mask-dir data/processed/masks --dump-worst 12 -o out
```

- `pct_ok` debería ser alto (> 95 %).
- `espejadas` > 0 indica fotos de ala izquierda (revisar).
- Los overlays de `out/worst/` muestran las máscaras con mayor disparidad Procrustes.

## 5. Reentrenar

Con el dataset combinado (upstream + Uruguay):

```bash
manganga train --data-dir . --epochs 40 --device cuda   # Colab/Kaggle
```

Para mantener ambos conjuntos, copiá las fotos y máscaras nuevas dentro de
`train/` y `masks/` con nombres únicos (por ejemplo `uy_0001.jpg`), o armá un
directorio `data/` propio con la misma estructura `train/ masks/ test/ test_masks/`
y pasalo con `--data-dir data/`.

## 6. Verificar

```bash
manganga batch -i data/raw/aladas -o out/ --model out/checkpoints/best.pt
```

Compará `out/report.csv` contra tus `.tps` manuales: la columna `disparity` debería
quedar por debajo del umbral (0.08 por defecto). Si no, revisá:

- resolución/iluminación de las fotos,
- que el ala sea la derecha,
- que el orden de los puntos manuales coincida con la referencia.
