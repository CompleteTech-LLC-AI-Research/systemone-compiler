param([string]$Title = 'System One benchmarks', [string]$Message = 'Benchmark status updated.')
$ErrorActionPreference = 'Stop'
$appId = 'CompleteTech.SystemOneBenchmarks'
$appKey = 'HKCU:\Software\Classes\AppUserModelId\' + $appId
New-Item -Path $appKey -Force | Out-Null
New-ItemProperty -Path $appKey -Name DisplayName -Value 'System One Benchmarks' -PropertyType String -Force | Out-Null
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$toastXml = New-Object Windows.Data.Xml.Dom.XmlDocument
$safeTitle = [System.Security.SecurityElement]::Escape($Title)
$safeMessage = [System.Security.SecurityElement]::Escape($Message)
$toastXml.LoadXml('<toast><visual><binding template="ToastGeneric"><text>' + $safeTitle + '</text><text>' + $safeMessage + '</text></binding></visual></toast>')
$toast = [Windows.UI.Notifications.ToastNotification]::new($toastXml)
$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId)
$setting = [string]$notifier.Setting
$notifier.Show($toast)
@{submitted=$true; notification_setting=$setting} | ConvertTo-Json -Compress
