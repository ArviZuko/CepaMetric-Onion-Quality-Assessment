$app = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyw = "C:\Program Files\Python313\pythonw.exe"
if (-not (Test-Path -LiteralPath $pyw)) { $pyw = "pythonw" }
$ico = Join-Path $app "SRC\app.ico"
$final = Join-Path $app "final.py"
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
function New-AppShortcut([string]$path) {
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = $pyw
    $shortcut.Arguments = '"' + $final + '"'
    $shortcut.WorkingDirectory = $app
    $shortcut.IconLocation = $ico
    $shortcut.Description = "Onion AI Inspection System"
    $shortcut.Save()
}
New-AppShortcut (Join-Path $desktop "Onion AI Inspection.lnk")
New-AppShortcut (Join-Path $app "Start Onion AI Inspection.lnk")
Write-Output "Desktop icon: $(Join-Path $desktop 'Onion AI Inspection.lnk')"