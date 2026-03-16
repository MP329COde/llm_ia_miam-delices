# Fichier: vision/caption.py — Génération de légende d'image via BLIP (Salesforce).
"""
Ce module charge le modèle BLIP (blip-image-captioning-base) de manière paresseuse
(lazy loading) et génère une description textuelle d'une image culinaire.

Il est conçu pour fonctionner sans modèle réel (mode stub) lorsque
`transformers` ou les poids BLIP ne sont pas disponibles, ce qui garantit la
testabilité sans GPU ni accès réseau.

Fonctions principales
---------------------
- is_blip_available() -> bool : vérifie si transformers/PIL sont importables.
- load_blip(model_id: str, device: str) -> tuple[processor, model] : charge le modèle.
- generate_caption(image_path: str, model_id: str, max_new_tokens: int) -> dict : produit la légende.

Exemple CLI
-----------
>>> python vision/caption.py --image data/sample/sample.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"

# Lazy imports — on ne crash pas si transformers est absent
try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    _BLIP_AVAILABLE = True
except Exception:
    BlipForConditionalGeneration = None  # type: ignore[assignment,misc]
    BlipProcessor = None  # type: ignore[assignment,misc]
    _BLIP_AVAILABLE = False

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except Exception:
    _PILImage = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except Exception:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def is_blip_available() -> bool:
    """Renvoie True si transformers, PIL et torch sont importables."""
    return _BLIP_AVAILABLE and _PIL_AVAILABLE and _TORCH_AVAILABLE


def select_device() -> str:
    """Sélectionne automatiquement cuda ou cpu."""
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Cache module-level pour éviter de recharger le modèle à chaque appel
_blip_cache: Dict[str, Any] = {}


def load_blip(
    model_id: str = DEFAULT_MODEL,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Charge le processeur et le modèle BLIP (mis en cache après le premier appel).

    Args:
        model_id: identifiant Hugging Face du modèle BLIP.
        device: 'cuda' ou 'cpu'. Si None, sélection automatique.

    Returns:
        tuple: (processor, model)

    Raises:
        ImportError: si transformers n'est pas installé.
        RuntimeError: si le chargement échoue.
    """
    if not _BLIP_AVAILABLE:
        raise ImportError(
            "Le module `transformers` est requis pour la génération de légendes. "
            "Installez-le avec : pip install transformers"
        )
    chosen_device = device or select_device()
    cache_key = f"{model_id}:{chosen_device}"
    if cache_key in _blip_cache:
        return _blip_cache[cache_key]
    try:
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(chosen_device)
        model.eval()
        _blip_cache[cache_key] = (processor, model)
        return processor, model
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger le modèle BLIP {model_id}: {exc}") from exc


def generate_caption(
    image_path: str,
    model_id: str = DEFAULT_MODEL,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Génère une légende textuelle pour une image.

    Args:
        image_path: chemin de l'image à analyser.
        model_id: modèle BLIP à utiliser.
        max_new_tokens: longueur maximale de la légende générée.
        device: 'cuda' ou 'cpu'. Si None, sélection automatique.

    Returns:
        dict avec les clés:
            - image: chemin de l'image
            - caption: texte généré
            - model: identifiant du modèle
            - device: périphérique utilisé

    Raises:
        FileNotFoundError: si l'image est introuvable.
        ImportError: si transformers n'est pas disponible.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    if not _PIL_AVAILABLE:
        raise ImportError("Pillow est requis pour ouvrir les images.")

    chosen_device = device or select_device()
    processor, model = load_blip(model_id=model_id, device=chosen_device)

    from PIL import Image as PILImage
    raw_image = PILImage.open(img_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(chosen_device)

    with _torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return {
        "image": str(img_path),
        "caption": caption,
        "model": model_id,
        "device": chosen_device,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(description="Génération de légende d'image (BLIP)")
    parser.add_argument("--image", required=True, help="Chemin de l'image.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Identifiant Hugging Face du modèle BLIP.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Nombre max de tokens générés."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée CLI."""
    import json
    args = parse_args(argv)
    result = generate_caption(args.image, model_id=args.model, max_new_tokens=args.max_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
