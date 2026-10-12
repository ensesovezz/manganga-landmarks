# Código heredado del upstream

Scripts originales de [Beeapp-landmark-detection](https://github.com/walterBSG/Beeapp-landmark-detection)
(Walter B. S. Gonçalves), conservados como referencia de reproducibilidad.

**No se usan en el pipeline nuevo** y **no funcionan** tal cual: tienen rutas
hardcodeadas a `/home/walter/...`, dependen de TensorFlow 1.x (`tf.Session`,
`tf.GraphDef`, `imp.load_source`) y de los modelos entrenados originales
(`.h5`, `.pb`) que nunca fueron publicados.

| Script | Qué hacía | Reemplazo en `manganga/` |
|---|---|---|
| `wing_detector.py` | Detección y recorte del ala con SSD MobileNet (TF1) | `crop.py` (segmentación clásica) |
| `cv2Detector.py` | Prueba del detector con `cv2.dnn` | `crop.py` |
| `unet_module.py` | Carga/evaluación de la U-Net en Keras | `models/unet.py`, `train.py` |
| `unet-test.py` | Evaluación de la U-Net | `train.evaluate_detection` |
| `BlobDetection.py` | Centros de los discos con `SimpleBlobDetector` | `landmarks/blobs.py` |
| `procrustes_module.py` | Procrustes global + ordenamiento heurístico | `landmarks/procrustes.py` |
| `myProcrustes.py` | Variante de Procrustes | `landmarks/procrustes.py` |
| `test_modules.py` | Test manual del pipeline completo | `tests/test_pipeline.py` |
| `total_process.py` | Pipeline completo por lotes | `cli.py` (`batch`, `train`) |
| `total_process_test_of_precision.py` | Medición de precisión de detección | `train.evaluate_detection` |

Los scripts que **sí** siguen vigentes en la raíz:

- `cria_mascara.py` — genera máscaras (discos) a partir de landmarks; se sigue usando
  para preparar datos nuevos (ver `docs/agregar-datos.md`).
- `utils.py`, `resize_square.py`, `rotate.py` — utilidades sueltas.
- `classifier.py` — utilidades de matriz de confusión (el clasificador de subespecies
  está fuera del alcance de este fork).
