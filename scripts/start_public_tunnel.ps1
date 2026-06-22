param(
    [int]$Port = 8000,
    [string]$LocalHost = "127.0.0.1",
    [int]$RemotePort = 80,
    [string]$RemoteUser = "nokey",
    [string]$RemoteHost = "localhost.run"
)

$ErrorActionPreference = "Stop"

$sshCommand = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $sshCommand) {
    throw "OpenSSH client was not found on PATH. Install the Windows OpenSSH client first."
}

$target = "${RemoteUser}@${RemoteHost}"
$remoteSpec = "${RemotePort}:${LocalHost}:${Port}"

Write-Host "Opening public tunnel for http://${LocalHost}:${Port}/ via ${RemoteHost}"
& $sshCommand.Source -T -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -R $remoteSpec $target

if ($LASTEXITCODE -ne 0) {
    throw "Failed to create public tunnel. ssh exited with code $LASTEXITCODE."
}