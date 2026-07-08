# Helper DX: attiva il virtualenv (se non gia' attivo) e lancia l'ingestion RAG.
# Uso:
#   .\run_ingest.ps1
#   .\run_ingest.ps1 ./data --title "Doc" --author "ACME"
#   .\run_ingest.ps1 ./data --title "Doc" --category "Tecnico" --description "..." --tags "sdcc,rag"
#   .\run_ingest.ps1 --sync  # <-- Nuovo: allinea il DB alla sorgente (elimina chunk orfani)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not $env:VIRTUAL_ENV) {
    $activate = Join-Path $root ".venv/Scripts/Activate.ps1"
    if (-not (Test-Path $activate)) {
        Write-Error "Virtualenv non trovato in '$activate'. Crealo con: python -m venv venv"
        exit 1
    }
    & $activate
}

python (Join-Path $root "scripts\ingest.py") @args
