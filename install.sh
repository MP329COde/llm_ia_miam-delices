#!/usr/bin/env bash
# Script d'installation local pour l'IA culinaire (CPU ou GPU).
set -euo pipefail

header() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

header "Initialisation de l'environnement virtuel (.venv)"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

header "Mise à jour de pip"
python -m pip install --upgrade pip

header "Détection du GPU (CUDA)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU NVIDIA détecté. CUDA devrait être disponible pour PyTorch."
else
  echo "Aucun GPU NVIDIA détecté ou nvidia-smi absent. Mode CPU-only."
fi

header "Installation des dépendances (requirements.txt)"
python -m pip install -r requirements.txt

header "Vérification PyTorch"
python - <<'PYCODE'
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
PYCODE

header "Installation terminée"
echo "Activez l'environnement avec: source .venv/bin/activate"
