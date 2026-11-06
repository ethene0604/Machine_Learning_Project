$ErrorActionPreference = 'Stop'
$results = New-Object System.Collections.Generic.List[object]

function Add-Result($kind, $name, $success, $detail) {
    $results.Add([pscustomobject]@{
        Kind = $kind
        Name = $name
        Success = $success
        Detail = $detail
    })
}

$tasks = Get-ScheduledTask | Where-Object {
    $_.TaskName -match 'NvDriverUpdateCheckDaily|NVIDIA App SelfUpdate|Office Automatic Updates|Office Feature Updates|BackgroundDownload|ScanForUpdates|Scheduled Start' -and
    ($_.TaskPath -eq '\' -or $_.TaskPath -match 'Office|VisualStudio|InstallService|WindowsUpdate') -and
    $_.State -ne 'Disabled'
}
foreach ($task in $tasks) {
    try {
        Disable-ScheduledTask -TaskName $task.TaskName -TaskPath $task.TaskPath -ErrorAction Stop | Out-Null
        Add-Result 'Task' ($task.TaskPath + $task.TaskName) $true 'Disabled'
    } catch {
        Add-Result 'Task' ($task.TaskPath + $task.TaskName) $false $_.Exception.Message
    }
}

$services = @('edgeupdate', 'edgeupdatem', 'MuseHubUpdaterService', 'OverwolfUpdater', 'WemeetUpdateSvc', 'wuauserv', 'UsoSvc')
foreach ($name in $services) {
    try {
        Set-Service -Name $name -StartupType Disabled -ErrorAction Stop
        Add-Result 'Service' $name $true 'Disabled startup'
    } catch {
        Add-Result 'Service' $name $false $_.Exception.Message
    }
}

$policies = @(
    @{ Path = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU'; Name = 'NoAutoUpdate'; Value = 1 },
    @{ Path = 'HKLM:\SOFTWARE\Policies\Microsoft\WindowsStore'; Name = 'AutoDownload'; Value = 2 },
    @{ Path = 'HKLM:\SOFTWARE\Policies\Microsoft\Office\16.0\Common\officeupdate'; Name = 'enableautomaticupdates'; Value = 0 }
)
foreach ($policy in $policies) {
    try {
        New-Item -Path $policy.Path -Force -ErrorAction Stop | Out-Null
        New-ItemProperty -Path $policy.Path -Name $policy.Name -Value $policy.Value -PropertyType DWord -Force -ErrorAction Stop | Out-Null
        Add-Result 'Policy' ($policy.Path + '\' + $policy.Name) $true ('Set ' + $policy.Value)
    } catch {
        Add-Result 'Policy' ($policy.Path + '\' + $policy.Name) $false $_.Exception.Message
    }
}

foreach ($name in @('wuauserv', 'UsoSvc')) {
    try {
        Stop-Service -Name $name -Force -ErrorAction Stop
        Add-Result 'RunningService' $name $true 'Stopped'
    } catch {
        Add-Result 'RunningService' $name $false $_.Exception.Message
    }
}

$results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath 'D:\My_projects\Machine_Learning_Project\auto_update_change_results_20261104.json' -Encoding UTF8
