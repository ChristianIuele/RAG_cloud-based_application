<#
.SYNOPSIS
    REDEPLOY dell'app SDCC RAG su un'Azure Web App for Containers GIA' esistente.

.DESCRIPTION
    Script di AGGIORNAMENTO IMMAGINE (non di provisioning). A differenza di
    infrastructure/deploy_azure.ps1 - che crea da zero l'intera infrastruttura
    (Resource Group, ACR, App Service Plan, Web App) - questo script NON crea
    nessuna risorsa: assume che ACR e Web App esistano gia' e le riusa.

    Cosa fa, in ordine:
        1. Determina il TAG da pubblicare:
           - se passato via -Tag, usa quello;
           - altrimenti legge l'ultimo tag versionato vX.Y.Z presente su ACR per il
             repository 'sdcc-rag', incrementa il PATCH di 1 (fallback v1.0.0 se non
             esiste alcun tag semver).
        2. Verifica che Docker Desktop sia attivo.
        3. Build senza cache -> login ACR -> tag -> push dell'immagine.
        4. Ripunta la Web App al nuovo tag (senza ripassare la registry url, per
           evitare il bug del DOPPIO PREFISSO login server - vedi nota sotto).
        5. Forza l'app setting VECTOR_STORE=azure_search e riavvia.
        6. Verifica finale: stato Running, nome immagine attivo (con controllo
           esplicito che il login server compaia UNA SOLA volta) e VECTOR_STORE.

    NOTA SUL BUG DEL DOPPIO PREFISSO (da evitare):
        Passare a `az webapp ... container set/create` un --container-image-name gia'
        comprensivo del login server INSIEME a --container-registry-url ha prodotto in
        passato un nome malformato del tipo:
            acrsdccrag2852.azurecr.io/acrsdccrag2852.azurecr.io/sdcc-rag:latest
        Qui il login server viene incluso UNA SOLA volta nel --container-image-name e
        la registry url NON viene ripassata (l'immagine e' gia' pushata e la Web App
        ha gia' le credenziali ACR configurate dal provisioning iniziale). Dopo il
        deploy lo script verifica che il login server compaia esattamente una volta.

.PARAMETER Tag
    Tag opzionale dell'immagine (formato vX.Y.Z). Se omesso, viene calcolato
    auto-incrementando il patch dell'ultimo tag presente su ACR.

.EXAMPLE
    .\redeploy_azure.ps1
    Redeploy con auto-increment del patch (es. se su ACR l'ultimo e' v1.0.1 -> v1.0.2).

.EXAMPLE
    .\redeploy_azure.ps1 -Tag v1.2.0
    Redeploy forzando esplicitamente il tag v1.2.0.

.NOTES
    Prerequisiti: Azure CLI (`az login` gia' effettuato), Docker Desktop attivo con
    accesso al Dockerfile nella stessa cartella di questo script. NON crea risorse
    Azure e NON modifica alcun file sorgente.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Tag
)

# Un fallimento di qualunque cmdlet ferma subito lo script; per i comandi nativi
# (docker/az) controlliamo esplicitamente $LASTEXITCODE (vedi sotto).
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# VARIABILI - risorse Azure GIA' esistenti (NON vengono create)
# ---------------------------------------------------------------------------
$ResourceGroup  = "rg-sdcc-rag"
$AcrName        = "acrsdccrag2852"
$AcrLoginServer = "acrsdccrag2852.azurecr.io"
$ImageName      = "sdcc-rag"
$WebAppName     = "app-sdcc-rag-7861"

# Ci posizioniamo nella cartella dello script (dove sta il Dockerfile), cosi' il
# build context "." e' sempre corretto a prescindere da dove viene lanciato.
Set-Location -Path $PSScriptRoot

# ---------------------------------------------------------------------------
# FUNZIONI DI SUPPORTO
# ---------------------------------------------------------------------------

# Calcola il prossimo tag vX.Y.(Z+1) leggendo i tag esistenti su ACR.
# Fallback a v1.0.0 se il repository non esiste o non ha tag semver.
function Get-NextPatchTag {
    param(
        [string]$AcrName,
        [string]$Repository
    )

    # La lettura dei tag NON e' fatale: un repository ancora inesistente e' un caso
    # legittimo (-> fallback), quindi la isoliamo in un suo try/catch.
    $raw = $null
    try {
        $raw = az acr repository show-tags --name $AcrName --repository $Repository --output json
        if ($LASTEXITCODE -ne 0) { $raw = $null }
    }
    catch {
        $raw = $null
    }

    if ([string]::IsNullOrWhiteSpace($raw)) {
        Write-Host "  Nessun tag leggibile su ACR per '$Repository': fallback a v1.0.0." -ForegroundColor Yellow
        return "v1.0.0"
    }

    $tags = $raw | ConvertFrom-Json

    # Teniamo solo i tag strettamente semver 'vX.Y.Z' e li ordiniamo per versione.
    $semver = @()
    foreach ($t in $tags) {
        if ($t -match '^v(\d+)\.(\d+)\.(\d+)$') {
            $semver += [pscustomobject]@{
                Tag     = $t
                Version = [version]("{0}.{1}.{2}" -f $Matches[1], $Matches[2], $Matches[3])
            }
        }
    }

    if ($semver.Count -eq 0) {
        Write-Host "  Nessun tag in formato vX.Y.Z su ACR: fallback a v1.0.0." -ForegroundColor Yellow
        return "v1.0.0"
    }

    $latest = ($semver | Sort-Object Version -Descending | Select-Object -First 1)
    $v = $latest.Version
    # In [version], .Build e' il terzo campo (il PATCH di vX.Y.Z).
    $next = "v{0}.{1}.{2}" -f $v.Major, $v.Minor, ($v.Build + 1)
    Write-Host "  Ultimo tag su ACR: $($latest.Tag) -> prossimo (patch +1): $next" -ForegroundColor Green
    return $next
}

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
Write-Host "=== REDEPLOY - SDCC RAG =================================" -ForegroundColor Cyan
Write-Host "Resource Group : $ResourceGroup" -ForegroundColor Cyan
Write-Host "ACR            : $AcrName ($AcrLoginServer)" -ForegroundColor Cyan
Write-Host "Web App        : $WebAppName" -ForegroundColor Cyan
Write-Host "Cartella       : $PSScriptRoot" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

try {
    # -----------------------------------------------------------------------
    # 0. DETERMINAZIONE DEL TAG
    # -----------------------------------------------------------------------
    Write-Host "`n[0/7] Determinazione del tag immagine..." -ForegroundColor Yellow
    if ([string]::IsNullOrWhiteSpace($Tag)) {
        Write-Host "  -Tag non fornito: calcolo automatico dall'ACR." -ForegroundColor Yellow
        $Tag = Get-NextPatchTag -AcrName $AcrName -Repository $ImageName
    }
    else {
        Write-Host "  -Tag fornito esplicitamente: $Tag" -ForegroundColor Green
    }
    Write-Host "  TAG SCELTO: $Tag" -ForegroundColor Green

    $localImage  = "$($ImageName):$($Tag)"
    $remoteImage = "$($AcrLoginServer)/$($ImageName):$($Tag)"

    # -----------------------------------------------------------------------
    # 1. VERIFICA DOCKER DESKTOP
    # -----------------------------------------------------------------------
    Write-Host "`n[1/7] Verifica Docker Desktop..." -ForegroundColor Yellow
    $dockerOk = $false
    try {
        $null = docker info 2>&1
        $dockerOk = ($LASTEXITCODE -eq 0)
    }
    catch {
        $dockerOk = $false
    }
    if (-not $dockerOk) {
        throw "Docker Desktop non risulta in esecuzione. Avvialo e rilancia lo script."
    }
    Write-Host "  Docker attivo." -ForegroundColor Green

    # -----------------------------------------------------------------------
    # 2. BUILD SENZA CACHE
    # -----------------------------------------------------------------------
    Write-Host "`n[2/7] Build immagine (--no-cache): $localImage ..." -ForegroundColor Yellow
    docker build --no-cache -t $localImage .
    if ($LASTEXITCODE -ne 0) { throw "docker build fallito (exit $LASTEXITCODE)." }
    Write-Host "  Build completata: $localImage" -ForegroundColor Green

    # -----------------------------------------------------------------------
    # 3. LOGIN ACR
    # -----------------------------------------------------------------------
    Write-Host "`n[3/7] Login sull'ACR '$AcrName'..." -ForegroundColor Yellow
    az acr login --name $AcrName
    if ($LASTEXITCODE -ne 0) { throw "az acr login fallito (exit $LASTEXITCODE)." }
    Write-Host "  Login ACR riuscito." -ForegroundColor Green

    # -----------------------------------------------------------------------
    # 4. TAG VERSO L'ACR
    # -----------------------------------------------------------------------
    Write-Host "`n[4/7] Tag: $localImage -> $remoteImage ..." -ForegroundColor Yellow
    docker tag $localImage $remoteImage
    if ($LASTEXITCODE -ne 0) { throw "docker tag fallito (exit $LASTEXITCODE)." }
    Write-Host "  Tag creato." -ForegroundColor Green

    # -----------------------------------------------------------------------
    # 5. PUSH
    # -----------------------------------------------------------------------
    Write-Host "`n[5/7] Push su $remoteImage (puo' richiedere qualche minuto)..." -ForegroundColor Yellow
    docker push $remoteImage
    if ($LASTEXITCODE -ne 0) { throw "docker push fallito (exit $LASTEXITCODE)." }
    Write-Host "  Immagine pubblicata su '$remoteImage'." -ForegroundColor Green

    # -----------------------------------------------------------------------
    # 6. RIPUNTA LA WEB APP + APP SETTING + RESTART
    # -----------------------------------------------------------------------
    Write-Host "`n[6/7] Aggiornamento Web App..." -ForegroundColor Yellow

    # 6a. Ripunta l'immagine. IMPORTANTE (bug doppio prefisso): il login server e' gia'
    #     dentro $remoteImage, quindi NON passiamo --container-registry-url ne'
    #     ripetiamo il login server altrove.
    Write-Host "  -> Container image: $remoteImage" -ForegroundColor Yellow
    az webapp config container set `
        --name $WebAppName `
        --resource-group $ResourceGroup `
        --container-image-name $remoteImage | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "az webapp config container set fallito (exit $LASTEXITCODE)." }
    Write-Host "  -> Web App ripuntata al nuovo tag." -ForegroundColor Green

    # 6b. App setting difensivo: forziamo il backend di ricerca cloud.
    #     (appsettings set provoca gia' un riavvio; il restart esplicito sotto e'
    #      comunque innocuo e assicura il pull della nuova immagine.)
    Write-Host "  -> App setting: VECTOR_STORE=azure_search" -ForegroundColor Yellow
    az webapp config appsettings set `
        --name $WebAppName `
        --resource-group $ResourceGroup `
        --settings VECTOR_STORE=azure_search | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "az webapp config appsettings set fallito (exit $LASTEXITCODE)." }
    Write-Host "  -> VECTOR_STORE impostato." -ForegroundColor Green

    # 6c. Restart esplicito.
    Write-Host "  -> Restart della Web App..." -ForegroundColor Yellow
    az webapp restart --name $WebAppName --resource-group $ResourceGroup
    if ($LASTEXITCODE -ne 0) { throw "az webapp restart fallito (exit $LASTEXITCODE)." }
    Write-Host "  -> Restart inviato." -ForegroundColor Green

    # -----------------------------------------------------------------------
    # 7. VERIFICA FINALE
    # -----------------------------------------------------------------------
    Write-Host "`n[7/7] Verifica finale..." -ForegroundColor Yellow

    # 7a. Stato dell'app.
    $state = az webapp show --name $WebAppName --resource-group $ResourceGroup --query "state" -o tsv
    if ($LASTEXITCODE -ne 0) { throw "az webapp show fallito (exit $LASTEXITCODE)." }
    if ($state -eq "Running") {
        Write-Host "  Stato app: $state" -ForegroundColor Green
    }
    else {
        Write-Host "  Stato app: $state (atteso 'Running')" -ForegroundColor Red
    }

    # 7b. Nome immagine attivo + controllo esplicito del DOPPIO PREFISSO.
    Write-Host "`n  Configurazione container attiva:" -ForegroundColor Yellow
    az webapp config container show --name $WebAppName --resource-group $ResourceGroup -o table
    if ($LASTEXITCODE -ne 0) { throw "az webapp config container show fallito (exit $LASTEXITCODE)." }

    $activeImage = az webapp config container show `
        --name $WebAppName `
        --resource-group $ResourceGroup `
        --query "[?name=='DOCKER_CUSTOM_IMAGE_NAME'].value | [0]" -o tsv
    if ($LASTEXITCODE -ne 0) { throw "lettura DOCKER_CUSTOM_IMAGE_NAME fallita (exit $LASTEXITCODE)." }

    Write-Host "`n  Immagine attiva grezza: $activeImage" -ForegroundColor Yellow
    # Conta quante volte il login server compare nel nome immagine finale.
    $loginServerCount = ([regex]::Matches($activeImage, [regex]::Escape($AcrLoginServer))).Count
    if ($loginServerCount -eq 1) {
        Write-Host "  OK: login server presente UNA sola volta (nessun doppio prefisso)." -ForegroundColor Green
    }
    elseif ($loginServerCount -gt 1) {
        throw "DOPPIO PREFISSO rilevato: il login server compare $loginServerCount volte in '$activeImage'."
    }
    else {
        Write-Host "  ATTENZIONE: login server non trovato nel nome immagine ('$activeImage')." -ForegroundColor Red
    }

    # Verifica che il tag attivo sia quello appena pubblicato.
    if ($activeImage -match [regex]::Escape(":$Tag") + '$') {
        Write-Host "  OK: tag attivo = $Tag" -ForegroundColor Green
    }
    else {
        Write-Host "  ATTENZIONE: il tag attivo non termina con ':$Tag'." -ForegroundColor Red
    }

    # 7c. VECTOR_STORE applicato.
    Write-Host "`n  App setting VECTOR_STORE:" -ForegroundColor Yellow
    az webapp config appsettings list `
        --name $WebAppName `
        --resource-group $ResourceGroup `
        --query "[?name=='VECTOR_STORE'].{Nome:name,Valore:value}" -o table
    if ($LASTEXITCODE -ne 0) { throw "az webapp config appsettings list fallito (exit $LASTEXITCODE)." }

    # -----------------------------------------------------------------------
    # FINE
    # -----------------------------------------------------------------------
    Write-Host "`n=== REDEPLOY COMPLETATO ================================" -ForegroundColor Green
    Write-Host "Tag pubblicato : $Tag" -ForegroundColor Green
    Write-Host "Immagine       : $remoteImage" -ForegroundColor Green
    Write-Host "URL            : https://$WebAppName.azurewebsites.net" -ForegroundColor Green
    Write-Host "(il primo avvio del nuovo container puo' richiedere alcuni minuti)" -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
}
catch {
    Write-Host "`n=== REDEPLOY FALLITO ===================================" -ForegroundColor Red
    Write-Host "Errore: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "========================================================" -ForegroundColor Red
    exit 1
}
