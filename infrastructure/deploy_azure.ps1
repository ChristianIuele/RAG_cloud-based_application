<#
.SYNOPSIS
    Deploy della Fase 3: pubblica l'immagine Docker dell'app (frontend Streamlit)
    su Azure Web App for Containers, con l'immagine ospitata su un Azure Container
    Registry dedicato.

.DESCRIPTION
    Orchestra il provisioning via Azure CLM (`az`):
        Resource Group -> ACR -> build/tag/push immagine -> App Service Plan (Linux)
        -> Web App for Containers.

    Topologia guidata dal principio della DATA GRAVITY: tutte le risorse di calcolo
    vanno in `germanywestcentral` per co-locarsi con Azure AI Search e Azure OpenAI,
    riducendo latenza ed egress cost.

    Lo script e' idempotente: ogni step verifica prima l'esistenza della risorsa
    ("check-then-create", come infrastructure/setup_azure_search_index.py) e crea
    solo se manca.

.NOTES
    Prerequisiti: Azure CLI (`az login` gia' effettuato), Docker con l'immagine
    locale `sdcc-rag:local` gia' costruita (vedi Dockerfile alla root del repo).
    I SEGRETI (chiavi API) NON sono in questo script: vedi il blocco commentato in
    fondo. Verranno iniettati al prossimo giro (Key Vault / pipeline), mai nel codice.
#>

# Un errore di qualunque comando az/docker ferma subito lo script (fail-fast).
#$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 1. VARIABILI
# ---------------------------------------------------------------------------

# Resource Group dedicato: separa il CALCOLO (questa app) dai DATI (Search/OpenAI,
# in un altro RG). Per riusare un RG esistente, sostituire questo valore col suo nome.
$ResourceGroup = "rg-sdcc-rag"

# Data Gravity: stessa region delle risorse dati per minimizzare latenza ed egress.
$Location = "germanywestcentral"

# Nome ACR: deve essere GLOBALMENTE univoco e senza trattini (regola di naming ACR).
# Il suffisso random garantisce l'unicita' al primo provisioning.
$AcrName = "acrsdccrag" + (Get-Random -Minimum 1000 -Maximum 9999)

# App Service Plan: nome stabile a livello di Resource Group (non globale).
$AppPlanName = "asp-sdcc-rag"

# Web App: l'hostname .azurewebsites.net e' globale -> serve un suffisso random.
$WebAppName = "app-sdcc-rag-" + (Get-Random -Minimum 1000 -Maximum 9999)

# Immagine locale di partenza (prodotta dal Dockerfile) da ri-taggare e pushare.
$ImageName = "sdcc-rag:local"

# NOTA SULL'IDEMPOTENZA:
# I suffissi Get-Random di $AcrName e $WebAppName cambiano a OGNI esecuzione, quindi
# rilanciare l'intero script creerebbe risorse NUOVE invece di riusare le esistenti.
# Le guardie "az ... show" qui sotto rendono idempotente ogni singolo step nella
# stessa esecuzione; per una vera idempotenza cross-run, dopo il primo giro FISSARE
# $AcrName e $WebAppName ai valori effettivamente creati (rimuovendo Get-Random).

Write-Host "=== Deploy Fase 3 - SDCC RAG =============================" -ForegroundColor Cyan
Write-Host "Resource Group : $ResourceGroup" -ForegroundColor Cyan
Write-Host "Location       : $Location" -ForegroundColor Cyan
Write-Host "ACR            : $AcrName" -ForegroundColor Cyan
Write-Host "App Plan       : $AppPlanName" -ForegroundColor Cyan
Write-Host "Web App        : $WebAppName" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 2. RESOURCE GROUP
# ---------------------------------------------------------------------------
Write-Host "`n[1/6] Resource Group..." -ForegroundColor Yellow

# `az group create` e' gia' idempotente; la guardia serve solo a un log piu' chiaro.
$rgExists = az group exists --name $ResourceGroup | ConvertFrom-Json
if ($rgExists) {
    Write-Host "  Resource Group '$ResourceGroup' gia' esistente: skip." -ForegroundColor Green
}
else {
    az group create --name $ResourceGroup --location $Location | Out-Null
    Write-Host "  Resource Group '$ResourceGroup' creato in '$Location'." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 3. AZURE CONTAINER REGISTRY (SKU Basic, admin abilitato)
# ---------------------------------------------------------------------------
Write-Host "`n[2/6] Azure Container Registry..." -ForegroundColor Yellow

# Guardia check-then-create: `az acr show` fallisce se l'ACR non esiste.
$acrExists = az acr show --name $AcrName --resource-group $ResourceGroup 2>$null
if ($acrExists) {
    Write-Host "  ACR '$AcrName' gia' esistente: skip." -ForegroundColor Green
}
else {
    # admin-enabled: espone user/password per il pull da parte della Web App
    # (semplice per la Fase 3; in futuro migrare a Managed Identity).
    az acr create `
        --name $AcrName `
        --resource-group $ResourceGroup `
        --location $Location `
        --sku Basic `
        --admin-enabled true | Out-Null
    Write-Host "  ACR '$AcrName' creato (SKU Basic, admin-enabled)." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 4. LOGIN + BUILD/TAG/PUSH DELL'IMMAGINE
# ---------------------------------------------------------------------------
Write-Host "`n[3/6] Push dell'immagine sull'ACR..." -ForegroundColor Yellow

# 4a. Login sull'ACR (usa il token dell'utente `az` gia' autenticato).
Write-Host "  -> Login sull'ACR '$AcrName'..." -ForegroundColor Yellow
az acr login --name $AcrName | Out-Null

# 4b. Recupera il login server (es. acrsdccrag1234.azurecr.io): prefisso dei tag remoti.
$loginServer = az acr show --name $AcrName --resource-group $ResourceGroup --query loginServer -o tsv
$remoteImage = "$loginServer/sdcc-rag:latest"
Write-Host "  -> Login server: $loginServer" -ForegroundColor Yellow

# 4c. Ritagga l'immagine locale con il path del registry remoto.
Write-Host "  -> Tag: $ImageName -> $remoteImage" -ForegroundColor Yellow
docker tag $ImageName $remoteImage

# 4d. Push sull'ACR.
Write-Host "  -> Push su $remoteImage (puo' richiedere qualche minuto)..." -ForegroundColor Yellow
docker push $remoteImage
Write-Host "  Immagine pubblicata su '$remoteImage'." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 5. APP SERVICE PLAN (Linux, SKU B1)
# ---------------------------------------------------------------------------
Write-Host "`n[4/6] App Service Plan..." -ForegroundColor Yellow

# NB: NIENTE piano F1 (Free). L'immagine Docker (~1.25 GB) su F1 andrebbe in OOM
# per i limiti di memoria del tier Free. B1 (Basic) offre RAM sufficiente al container.
$planExists = az appservice plan show --name $AppPlanName --resource-group $ResourceGroup 2>$null
if ($planExists) {
    Write-Host "  App Service Plan '$AppPlanName' gia' esistente: skip." -ForegroundColor Green
}
else {
    az appservice plan create `
        --name $AppPlanName `
        --resource-group $ResourceGroup `
        --location $Location `
        --is-linux `
        --sku B1 | Out-Null
    Write-Host "  App Service Plan '$AppPlanName' creato (Linux, B1)." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 6. WEB APP FOR CONTAINERS
# ---------------------------------------------------------------------------
Write-Host "`n[5/6] Web App for Containers..." -ForegroundColor Yellow

$webappExists = az webapp show --name $WebAppName --resource-group $ResourceGroup 2>$null
if ($webappExists) {
    Write-Host "  Web App '$WebAppName' gia' esistente: skip creazione." -ForegroundColor Green
}
else {
    # Credenziali admin dell'ACR: servono alla Web App per fare il pull dell'immagine.
    $acrUser = az acr credential show --name $AcrName --query username -o tsv
    $acrPass = az acr credential show --name $AcrName --query "passwords[0].value" -o tsv

    az webapp create `
        --name $WebAppName `
        --resource-group $ResourceGroup `
        --plan $AppPlanName `
        --container-image-name $remoteImage `
        --container-registry-url "https://$loginServer" `
        --container-registry-user $acrUser `
        --container-registry-password $acrPass | Out-Null
    Write-Host "  Web App '$WebAppName' creata e agganciata a '$remoteImage'." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 6b. CONFIG NON-SEGRETA: porta del container
# ---------------------------------------------------------------------------
Write-Host "`n[6/6] Configurazione porta..." -ForegroundColor Yellow

# Il Dockerfile avvia Streamlit su 8501, ma la Web App for Containers instrada di
# default verso 80/8080: senza WEBSITES_PORT il sito resta irraggiungibile.
# NON e' un segreto -> lo impostiamo qui (a differenza delle chiavi API, vedi sotto).
az webapp config appsettings set `
    --name $WebAppName `
    --resource-group $ResourceGroup `
    --settings WEBSITES_PORT=8501 | Out-Null
Write-Host "  WEBSITES_PORT=8501 impostata." -ForegroundColor Green

Write-Host "`n=== Deploy completato ====================================" -ForegroundColor Green
Write-Host "URL: https://$WebAppName.azurewebsites.net" -ForegroundColor Green
Write-Host "(il primo avvio del container puo' richiedere alcuni minuti)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green


# ===========================================================================
# PROSSIMO GIRO - INIEZIONE DEI SEGRETI (NON in questo commit)
# ===========================================================================
# PROMEMORIA PER IL TECH LEAD:
# I segreti applicativi (chiavi API) NON vanno passati per il codice ne' committati.
# Al prossimo giro verranno iniettati come app settings recuperandoli da una fonte
# sicura (Azure Key Vault reference oppure variabili segrete della pipeline CI/CD).
#
# Array delle chiavi attese dall'app (i VALORI restano fuori dal repo):
#
# $AppSettings = @(
#     "OPENAI_API_KEY=<da-key-vault>",
#     "AI_SEARCH_KEY=<da-key-vault>",
#     "AZURE_OPENAI_API_KEY=<da-key-vault>",
#     "AZURE_SEARCH_ADMIN_KEY=<da-key-vault>"
# )
#
# az webapp config appsettings set `
#     --name $WebAppName `
#     --resource-group $ResourceGroup `
#     --settings $AppSettings
# ===========================================================================
