# CrohnSight - build/test or push the Docker image on Windows (Docker Desktop must be running).
#   docker_build_test.bat  ->  builds hilkalathiya59/crohnsight:1.0 and test-runs it on port 8765
#   docker_push.bat        ->  pushes it to Docker Hub (create the PRIVATE repo "crohnsight" on hub.docker.com first)
# Every step is also written to docker_<mode>_log.txt next to this file.
param([ValidateSet('build', 'push')][string]$Mode = 'build')
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$Image = 'hilkalathiya59/crohnsight:1.0'
$Log = Join-Path $PSScriptRoot ("docker_{0}_log.txt" -f $Mode)
Set-Content -Path $Log -Value ("CrohnSight docker {0} - {1}" -f $Mode, (Get-Date))

function Log([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $Log -Value $line
}
function Run([string]$exe, [string[]]$argv) {
    & $exe @argv 2>&1 | ForEach-Object { $s = "$_"; Write-Host $s; Add-Content -Path $Log -Value $s }
    return $LASTEXITCODE
}
function Finish([string]$result, [int]$code) {
    Log ("RESULT: " + $result)
    Read-Host 'Finished. Press Enter to close this window'
    exit $code
}

if ((Run 'docker' @('version', '--format', 'Client {{.Client.Version}} / Server {{.Server.Version}}')) -ne 0) {
    Log 'Docker is not running. Start Docker Desktop, wait until it shows "Engine running", then run this again.'
    Finish 'FAILED (Docker not running)' 1
}

if ($Mode -eq 'build') {
    Log "Building $Image (the first build downloads about 250 MB and takes 5-10 minutes) ..."
    if ((Run 'docker' @('compose', 'build')) -ne 0) { Finish 'FAILED (build error, see the lines above)' 1 }
    Log 'Build done. Test run on http://localhost:8765 ...'
    & docker rm -f crohnsight-test 2>&1 | Out-Null
    Run 'docker' @('run', '-d', '--name', 'crohnsight-test', '-p', '8765:8000', $Image) | Out-Null
    $ok = $false
    for ($i = 0; $i -lt 60 -and -not $ok; $i++) {
        Start-Sleep -Seconds 2
        try {
            $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 3
            if ($h.ok) { $ok = $true; Log ('Health: ' + ($h | ConvertTo-Json -Compress)) }
        } catch { }
    }
    if ($ok) {
        foreach ($p in @('/', '/capsule/', '/video/', '/api/video/local')) {
            try { $r = Invoke-WebRequest -Uri ('http://127.0.0.1:8765' + $p) -UseBasicParsing -TimeoutSec 10; Log ("GET {0} -> {1}" -f $p, $r.StatusCode) }
            catch { Log ("GET {0} -> FAILED {1}" -f $p, $_.Exception.Message); $ok = $false }
        }
    }
    Run 'docker' @('logs', 'crohnsight-test') | Out-Null
    & docker rm -f crohnsight-test 2>&1 | Out-Null
    Run 'docker' @('image', 'ls', $Image) | Out-Null
    if ($ok) { Finish 'BUILD AND TEST OK' 0 } else { Finish 'TEST FAILED (see the container log above)' 1 }
}
else {
    # Docker Hub answers this address without a login only for PUBLIC repositories (private or missing -> 404).
    $HubUrl = 'https://hub.docker.com/v2/repositories/hilkalathiya59/crohnsight/'
    function Test-RepoPublic {
        try { $r = Invoke-RestMethod -Uri $HubUrl -TimeoutSec 20; return ($r.is_private -eq $false) } catch { return $false }
    }
    if (Test-RepoPublic) {
        Log 'hilkalathiya59/crohnsight is PUBLIC on Docker Hub. Open it on hub.docker.com > Settings > Change visibility > Private, then run this again.'
        Finish 'STOPPED (repository is public, nothing pushed)' 1
    }
    Log "Pushing $Image to Docker Hub ..."
    if ((Run 'docker' @('push', $Image)) -ne 0) {
        Log 'Push failed. If it says "denied" or "unauthorized", sign in to Docker Desktop as hilkalathiya59 and run this again.'
        Finish 'FAILED (push)' 1
    }
    Start-Sleep -Seconds 5
    if (Test-RepoPublic) {
        Log 'WARNING: the repository is PUBLIC. Make it private NOW: hub.docker.com > hilkalathiya59/crohnsight > Settings > Change visibility > Private.'
        Finish 'PUSHED BUT PUBLIC - make it private now' 1
    }
    Log 'Checked: the repository is not publicly visible.'
    Finish 'PUSH OK (private)' 0
}
