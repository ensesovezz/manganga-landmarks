"""Interfaz de línea de comandos de manganga.

    manganga detect -i ala.jpg -o out/
    manganga batch -i fotos/ -o out/
    manganga train --data-dir . --epochs 40
    manganga validate-data --mask-dir masks
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from manganga import N_LANDMARKS, __version__
from manganga.landmarks.procrustes import DEFAULT_MAX_DISPARITY

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Detección de 19 landmarks en alas de Apis mellifera (esquema Ruttner).",
)
console = Console()

DEFAULT_CHECKPOINT = Path("out/checkpoints/best.pt")


def _load_runner(model_path: Path, device: str, max_disparity: float, use_clahe: Optional[bool]):
    from manganga.detect import DetectionRunner, load_checkpoint

    model, config = load_checkpoint(model_path, device=device)
    return DetectionRunner(
        model,
        config=config,
        device=device,
        max_disparity=max_disparity,
        use_clahe=use_clahe,
    )


def _write_outputs(result, out_dir: Path, prepared_image=None) -> None:
    from manganga.io import save_overlay, write_csv, write_tps

    if result.points is None:
        return
    stem = Path(result.image).stem
    write_tps(out_dir / "landmarks" / f"{stem}.tps", result.points, image_id=result.image)
    write_csv(out_dir / "landmarks" / f"{stem}.csv", result.points)
    if prepared_image is not None:
        save_overlay(out_dir / "overlays" / f"{stem}.png", prepared_image, result.points)


@app.command()
def detect(
    image: Path = typer.Option(..., "-i", "--image", exists=True, dir_okay=False, help="Foto del ala."),
    out: Path = typer.Option(Path("out"), "-o", "--out", help="Directorio de salida."),
    model: Path = typer.Option(DEFAULT_CHECKPOINT, "-m", "--model", help="Checkpoint de la U-Net."),
    crop: bool = typer.Option(False, "--crop", help="Recortar el ala automáticamente (fotos crudas)."),
    no_clahe: bool = typer.Option(False, "--no-clahe", help="Desactivar CLAHE."),
    max_disparity: float = typer.Option(DEFAULT_MAX_DISPARITY, help="Umbral de disparidad Procrustes."),
    device: str = typer.Option("cpu", help="cpu o cuda."),
) -> None:
    """Detecta los 19 landmarks en una sola imagen."""
    from manganga.detect import collect_images  # noqa: F401
    from manganga.preprocess import apply_clahe, letterbox, load_gray

    runner = _load_runner(model, device, max_disparity, None if not no_clahe else False)
    result = runner.detect(image, crop=crop)

    prepared = None
    try:
        from manganga.crop import crop_wing

        raw = load_gray(image)
        prepared = crop_wing(raw, size=runner.size).image if crop else letterbox(raw, size=runner.size)
        if prepared is not None and runner.use_clahe:
            prepared = apply_clahe(prepared)
    except Exception:  # el overlay es opcional
        prepared = None

    out.mkdir(parents=True, exist_ok=True)
    _write_outputs(result, out, prepared_image=prepared)

    table = Table(title=f"manganga detect · {image.name}")
    table.add_column("campo")
    table.add_column("valor")
    for key, value in result.as_row().items():
        table.add_row(str(key), str(value))
    console.print(table)

    if not result.ok:
        console.print(
            f"[yellow]No se pudieron obtener los {N_LANDMARKS} puntos.[/yellow] "
            "Probá con --crop, revisá la calidad de la foto o el checkpoint usado."
        )
        raise typer.Exit(code=1)
    console.print(f"[green]Listo:[/green] {out}/landmarks/{Path(image).stem}.tps")


@app.command()
def batch(
    input_dir: Path = typer.Option(..., "-i", "--input-dir", exists=True, file_okay=False),
    out: Path = typer.Option(Path("out"), "-o", "--out"),
    model: Path = typer.Option(DEFAULT_CHECKPOINT, "-m", "--model"),
    crop: bool = typer.Option(False, "--crop", help="Recortar cada ala automáticamente."),
    no_clahe: bool = typer.Option(False, "--no-clahe"),
    max_disparity: float = typer.Option(DEFAULT_MAX_DISPARITY),
    device: str = typer.Option("cpu"),
    limit: Optional[int] = typer.Option(None, help="Procesar solo las primeras N imágenes."),
) -> None:
    """Procesa un directorio completo y escribe un reporte CSV."""
    import csv

    from manganga.detect import collect_images
    from manganga.io import save_overlay, write_csv, write_tps
    from manganga.preprocess import apply_clahe, letterbox, load_gray

    paths = collect_images(input_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        console.print(f"[red]No hay imágenes en {input_dir}[/red]")
        raise typer.Exit(code=1)

    runner = _load_runner(model, device, max_disparity, None if not no_clahe else False)
    results = runner.detect_batch(paths, crop=crop)

    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for path, result in zip(paths, results):
        prepared = None
        if result.ok:
            try:
                from manganga.crop import crop_wing

                raw = load_gray(path)
                prepared = crop_wing(raw, size=runner.size).image if crop else letterbox(raw, size=runner.size)
                if prepared is not None and runner.use_clahe:
                    prepared = apply_clahe(prepared)
            except Exception:
                prepared = None
        _write_outputs(result, out, prepared_image=prepared)
        ok += int(result.ok)

    report = out / "report.csv"
    with open(report, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].as_row().keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_row())

    console.print(
        f"[green]{ok}/{len(results)}[/green] imágenes con los {N_LANDMARKS} puntos "
        f"({100 * ok / len(results):.1f}%). Reporte: {report}"
    )


@app.command()
def train(
    data_dir: Path = typer.Option(Path("."), "--data-dir", help="Directorio con train/ masks/ test/ test_masks/."),
    out: Path = typer.Option(Path("out/checkpoints"), "-o", "--out"),
    epochs: int = typer.Option(40),
    batch_size: int = typer.Option(8, "--batch-size"),
    size: int = typer.Option(400),
    base: int = typer.Option(32, help="Canales de la primera capa (16/24/32)."),
    depth: int = typer.Option(4, help="Cantidad de niveles de downsampling."),
    lr: float = typer.Option(1e-3),
    pos_weight: float = typer.Option(50.0, help="Peso de los píxeles positivos (igual que el upstream)."),
    val_split: float = typer.Option(0.05),
    max_train_samples: Optional[int] = typer.Option(None, help="Limitar el set de entrenamiento (smoke test)."),
    metric_every: int = typer.Option(5, help="Cada cuántas épocas correr la métrica completa."),
    metric_samples: int = typer.Option(200, help="Imágenes usadas en la métrica completa."),
    workers: int = typer.Option(0),
    seed: int = typer.Option(0),
    max_disparity: float = typer.Option(DEFAULT_MAX_DISPARITY),
    device: str = typer.Option("cpu"),
) -> None:
    """Entrena la U-Net (en CPU conviene --base 16 --size 256 --max-train-samples 500)."""
    from manganga.train import TrainConfig, train as run_train

    config = TrainConfig(
        data_dir=data_dir,
        out_dir=out,
        size=size,
        base=base,
        depth=depth,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        pos_weight=pos_weight,
        val_split=val_split,
        max_train_samples=max_train_samples,
        metric_every=metric_every,
        metric_samples=metric_samples,
        workers=workers,
        seed=seed,
        max_disparity=max_disparity,
        device=device,
    )
    summary = run_train(config)
    console.print(f"[green]Mejor checkpoint:[/green] {summary['best_checkpoint']}")
    console.print(f"Métrica ({N_LANDMARKS} puntos ok): {summary['best_score_pct_ok_ordenados']}%")


@app.command("validate-data")
def validate_data(
    mask_dir: Path = typer.Option(Path("masks"), "--mask-dir", exists=True, file_okay=False),
    out: Path = typer.Option(Path("out"), "-o", "--out"),
    limit: Optional[int] = typer.Option(None),
    max_disparity: float = typer.Option(DEFAULT_MAX_DISPARITY),
    dump_worst: int = typer.Option(0, help="Guardar overlays de las N peores máscaras."),
) -> None:
    """Valida el ground truth: extrae los discos, los ordena y mide la disparidad."""
    from manganga.data.validate import dump_worst as dump_worst_fn
    from manganga.data.validate import summarize, validate_dataset

    df = validate_dataset(mask_dir, limit=limit, max_disparity=max_disparity)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"validation_{mask_dir.name}.csv"
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    table = Table(title=f"Validación de {mask_dir}")
    table.add_column("métrica")
    table.add_column("valor", justify="right")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"Detalle: {csv_path}")

    if dump_worst:
        paths = dump_worst_fn(df, mask_dir, out / "worst", n=dump_worst)
        console.print(f"Overlays de las peores {len(paths)} en {out / 'worst'}")


@app.command()
def version() -> None:
    """Muestra la versión."""
    console.print(f"manganga {__version__}")


if __name__ == "__main__":
    app()
