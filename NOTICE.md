# Atribución y alcance de la licencia

## Trabajo original

- **Repo**: https://github.com/walterBSG/Beeapp-landmark-detection
- **Autor**: Walter B. S. Gonçalves
- **Paper**:

  > Gonçalves, W. B. S., et al. *Deep Learning and Geometric Morphometrics for the
  > Identification of Honey Bee Subspecies*. Big Data and Cognitive Computing, 6(3), 70, 2022.
  > https://www.mdpi.com/2504-2289/6/3/70

El pipeline de morfometría geométrica, el esquema de 19 landmarks, el dataset
(`train/`, `masks/`, `test/`, `test_masks/`) y los scripts originales provienen de
ese repositorio.

## Licencia

El repositorio original **no incluye ningún archivo de licencia** (`licenseInfo: null`
en la API de GitHub), por lo que, por defecto, sus contenidos quedan bajo
"todos los derechos reservados".

Por lo tanto:

- La licencia **MIT** de este fork (`LICENSE`) aplica **solamente** al código nuevo:
  el paquete `manganga/`, los scripts agregados, los tests, los notebooks y la
  documentación.
- El material heredado (scripts originales en la raíz, dataset, máscaras, modelos
  `.joblib`) se conserva **con fines de investigación y reproducibilidad**, con
  atribución explícita al autor original.
- Si el autor original publica una licencia o pide cambios, este repo se ajustará.

## Contribuciones de este fork

- Portabilidad a Python 3.10-3.12 y PyTorch (el original usaba TensorFlow 1.x).
- Reentrenamiento de la U-Net (la arquitectura y los pesos originales no fueron publicados).
- Ordenamiento de landmarks por asignación óptima (Hungarian + Procrustes) en lugar de
  heurísticas con swaps manuales.
- Detección de puntos con fallbacks y reporte de estado (el original descartaba imágenes
  en silencio).
- Recorte clásico del ala, sin depender del detector SSD de TensorFlow.
- CLI, tests y documentación en español.
