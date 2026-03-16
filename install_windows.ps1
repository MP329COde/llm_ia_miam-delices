# install_windows.ps1 — installation locale pour Windows (CPU ou GPU CUDA)
# Usage :
#   powershell -ExecutionPolicy Bypass -File .\install_windows.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Header($msg) {
    Write-Host "============================================================"
    Write-Host $msg
    Write-Host "============================================================"
}

Write-Header "Création de l'environnement virtuel (.venv)"
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Header "Activation de l'environnement"
& ".\.venv\Scripts\Activate.ps1"

Write-Header "Mise à jour de pip"
python -m pip install --upgrade pip

Write-Header "Détection GPU (CUDA) si disponible"
try {
    $nvidia = & nvidia-smi
    Write-Host "GPU détecté, sortie nvidia-smi :"
    Write-Host $nvidia
} catch {
    Write-Warning "nvidia-smi introuvable : installation CPU-only (ok)."
}

Write-Header "Installation des dépendances (requirements.txt)"
python -m pip install -r requirements.txt

Write-Header "Vérification PyTorch"
python -c "import torch;print(f'PyTorch version: {torch.__version__}');print(f'CUDA available: {torch.cuda.is_available()}');print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'GPU: none')"

Write-Header "Installation terminée"
Write-Host "Activez l'environnement avec: .\\.venv\\Scripts\\Activate.ps1"
