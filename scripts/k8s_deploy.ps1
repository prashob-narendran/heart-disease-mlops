# Deploy the heart-disease API to the current Kubernetes context and verify it.
#
#   .\scripts\k8s_deploy.ps1                       # deploy :latest from GHCR
#   .\scripts\k8s_deploy.ps1 -Tag sha-c6c4a6c      # deploy a specific image tag
#   .\scripts\k8s_deploy.ps1 -Local                # build heart-api:local and deploy it
#   .\scripts\k8s_deploy.ps1 -Url http://localhost:8080
#
# Steps: [build] -> apply manifests (kustomize) -> optionally pin the tag ->
# wait for the rollout -> show pods/service/HPA -> smoke-test the service.
param(
    [string]$Tag = "",
    [string]$Url = "http://localhost:8080",
    [switch]$Local
)
$ErrorActionPreference = "Stop"
$NS = "heart-disease"
$Image = "ghcr.io/prashob-narendran/heart-disease-api"
Set-Location (Join-Path $PSScriptRoot "..")

function Invoke-Step([string]$Title, [scriptblock]$Block) {
    Write-Host "== $Title" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Title" }
}

Invoke-Step "Context: $(kubectl config current-context)" { kubectl get nodes }
if ($Local) {
    Invoke-Step "Building heart-api:local" { docker build -t heart-api:local . }
    Invoke-Step "Applying manifests (kubectl apply -k k8s/local/)" {
        kubectl apply -k k8s/local/
    }
    # same tag as before -> force new pods so they run the fresh build
    Invoke-Step "Restarting pods" {
        kubectl -n $NS rollout restart deployment/heart-api
    }
} else {
    Invoke-Step "Applying manifests (kubectl apply -k k8s/)" { kubectl apply -k k8s/ }
}

if ($Tag -and -not $Local) {
    Invoke-Step "Pinning image to ${Image}:$Tag" {
        kubectl -n $NS set image deployment/heart-api "api=${Image}:$Tag"
    }
}

Invoke-Step "Waiting for rollout" {
    kubectl -n $NS rollout status deployment/heart-api --timeout=180s
}
Invoke-Step "Resources" { kubectl -n $NS get deploy,pods,svc,hpa,pdb -o wide }
Invoke-Step "Smoke test against $Url" {
    python scripts/smoke_test_api.py --url $Url --wait 60
}
Write-Host "Deployment verified." -ForegroundColor Green
