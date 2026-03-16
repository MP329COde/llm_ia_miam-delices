# Fichier: data/import_images.py — Import d'images depuis URL, dossier ou ZIP.
"""
Ce module fournit un pipeline d'import d'images pour constituer ou enrichir
un dataset au format YOLO.

Fonctionnalités
---------------
- Import depuis un **dossier local** : copie/redimensionne les images trouvées.
- Import depuis un **fichier ZIP** : extrait les images dans la structure cible.
- Import depuis une **liste d'URLs** : télécharge les images de manière robuste.
- Redimensionnement optionnel (PIL) et détection du format.
- Découpe automatique train/val selon un ratio configurable.
- Log de chaque opération dans un fichier `import_log.json`.

Structure de sortie
-------------------
    dataset/
    ├── images/
    │   ├── train/   ← (1 - val_split) des images
    │   └── val/     ← val_split des images
    └── import_log.json

Exemple CLI
-----------
>>> python data/import_images.py --source monDossier/ --output-dir data/dataset
>>> python data/import_images.py --source archive.zip  --output-dir data/dataset
>>> python data/import_images.py --urls urls.txt       --output-dir data/dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Extensions d'image reconnues
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _md5_short(path: Path) -> str:
    """Calcule un hash MD5 court du contenu d'un fichier (8 caractères)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _safe_stem(path: Path, existing: set) -> str:
    """
    Retourne un nom de fichier unique (sans extension) pour éviter les collisions.
    Ajoute le hash MD5 court si nécessaire.
    """
    stem = path.stem
    if stem not in existing:
        return stem
    return f"{stem}_{_md5_short(path)}"


def resize_image(
    src: Path,
    dst: Path,
    max_size: Optional[int],
) -> None:
    """
    Copie l'image de src vers dst, en la redimensionnant si max_size est défini.

    Args:
        src: chemin source.
        dst: chemin destination.
        max_size: dimension maximale (largeur ou hauteur). None = pas de redim.
    """
    if max_size is None or not _PIL_AVAILABLE:
        shutil.copy2(src, dst)
        return
    with _PILImage.open(src) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, _PILImage.LANCZOS)
        img.save(dst, format="JPEG", quality=92)


def split_train_val(
    files: List[Path],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Découpe une liste de fichiers en ensembles train et val.

    Args:
        files: liste de chemins.
        val_split: proportion de validation (0-1).
        seed: graine aléatoire pour la reproductibilité.

    Returns:
        (train_files, val_files)
    """
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Sources d'import
# ---------------------------------------------------------------------------

def import_from_folder(
    source_dir: Path,
    output_dir: Path,
    val_split: float = 0.2,
    max_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Importe les images d'un dossier local vers la structure train/val.

    Args:
        source_dir: dossier contenant les images.
        output_dir: dossier de sortie.
        val_split: ratio de validation.
        max_size: redimensionnement maximal (px).
        seed: graine aléatoire.

    Returns:
        dict de logs (nombre d'images copiées, erreurs, etc.)
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Dossier source introuvable: {source_dir}")

    image_files = sorted(p for p in source_dir.rglob("*") if _is_image_file(p))
    return _process_images(image_files, output_dir, val_split, max_size, seed)


def import_from_zip(
    zip_path: Path,
    output_dir: Path,
    val_split: float = 0.2,
    max_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Extrait et importe les images d'une archive ZIP.

    Args:
        zip_path: chemin du fichier ZIP.
        output_dir: dossier de sortie.
        val_split: ratio de validation.
        max_size: redimensionnement maximal (px).
        seed: graine aléatoire.

    Returns:
        dict de logs.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive ZIP introuvable: {zip_path}")

    tmp_dir = output_dir / "_zip_extract_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        return import_from_folder(tmp_dir, output_dir, val_split, max_size, seed)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def import_from_urls(
    urls: List[str],
    output_dir: Path,
    val_split: float = 0.2,
    max_size: Optional[int] = None,
    seed: int = 42,
    timeout: int = 30,
) -> dict:
    """
    Télécharge et importe des images depuis une liste d'URLs.

    Args:
        urls: liste d'URLs d'images.
        output_dir: dossier de sortie.
        val_split: ratio de validation.
        max_size: redimensionnement maximal (px).
        seed: graine aléatoire.
        timeout: délai d'attente par URL en secondes.

    Returns:
        dict de logs.
    """
    tmp_dir = output_dir / "_url_download_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    errors: List[str] = []

    for i, url in enumerate(urls):
        url = url.strip()
        if not url:
            continue
        try:
            req = Request(url, headers={"User-Agent": "miam-delices/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            # Déduire l'extension depuis l'URL ou Content-Type
            suffix = Path(url.split("?")[0]).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                suffix = ".jpg"
            fname = tmp_dir / f"img_{i:04d}{suffix}"
            fname.write_bytes(data)
            downloaded.append(fname)
        except (URLError, Exception) as exc:
            errors.append(f"{url}: {exc}")
            logger.warning("Téléchargement échoué: %s (%s)", url, exc)

    try:
        result = _process_images(downloaded, output_dir, val_split, max_size, seed)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result["download_errors"] = errors
    result["downloaded"] = len(downloaded)
    return result


# ---------------------------------------------------------------------------
# Traitement commun
# ---------------------------------------------------------------------------

def _process_images(
    image_files: List[Path],
    output_dir: Path,
    val_split: float,
    max_size: Optional[int],
    seed: int,
) -> dict:
    """Traitement interne commun : split, copie, log."""
    output_dir = Path(output_dir)
    train_dir = output_dir / "images" / "train"
    val_dir = output_dir / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if not image_files:
        return {"train": 0, "val": 0, "errors": [], "total": 0}

    train_files, val_files = split_train_val(image_files, val_split=val_split, seed=seed)

    log = {"train": 0, "val": 0, "errors": [], "files": []}
    existing_names: set = set()

    def _copy(src: Path, dst_dir: Path, split: str) -> None:
        stem = _safe_stem(src, existing_names)
        existing_names.add(stem)
        dst = dst_dir / f"{stem}{src.suffix.lower()}"
        try:
            resize_image(src, dst, max_size)
            log[split] += 1  # type: ignore[literal-required]
            log["files"].append({"src": str(src), "dst": str(dst), "split": split})
        except Exception as exc:
            log["errors"].append({"src": str(src), "error": str(exc)})

    for f in train_files:
        _copy(f, train_dir, "train")
    for f in val_files:
        _copy(f, val_dir, "val")

    log["total"] = log["train"] + log["val"]

    # Sauvegarder le log
    log_path = output_dir / "import_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Import terminé: %d train, %d val, %d erreurs.", log["train"], log["val"], len(log["errors"]))
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import d'images dans la structure train/val pour YOLOv8."
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--source", type=Path, help="Dossier local contenant les images.")
    src_group.add_argument("--zip", type=Path, help="Archive ZIP contenant les images.")
    src_group.add_argument("--urls", type=Path, help="Fichier texte contenant une URL par ligne.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dossier de sortie.")
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Ratio de validation (défaut: 0.2)."
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Dimension maximale des images (px). Si omis, pas de redimensionnement.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour le split.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.source:
        result = import_from_folder(
            args.source, args.output_dir,
            val_split=args.val_split, max_size=args.max_size, seed=args.seed
        )
    elif args.zip:
        result = import_from_zip(
            args.zip, args.output_dir,
            val_split=args.val_split, max_size=args.max_size, seed=args.seed
        )
    elif args.urls:
        urls = Path(args.urls).read_text(encoding="utf-8").splitlines()
        result = import_from_urls(
            urls, args.output_dir,
            val_split=args.val_split, max_size=args.max_size, seed=args.seed
        )
    else:
        raise SystemExit("Spécifiez --source, --zip ou --urls.")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
