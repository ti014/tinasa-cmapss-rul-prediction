param(
    [string]$Deck = (Join-Path $PSScriptRoot '..\slides\cmapss_fd001_presentation.pptx'),
    [string]$PreviewDirectory = (Join-Path $PSScriptRoot '..\tmp\slide-preview-20260929')
)

$ErrorActionPreference = 'Stop'
$deckPath = (Resolve-Path -LiteralPath $Deck).Path
$pdfPath = [System.IO.Path]::ChangeExtension($deckPath, '.pdf')
$previewPath = [System.IO.Path]::GetFullPath($PreviewDirectory)
[System.IO.Directory]::CreateDirectory($previewPath) | Out-Null
$application = New-Object -ComObject PowerPoint.Application
$presentation = $null
try {
    # Open without a visible window; do not touch other open presentations.
    $presentation = $application.Presentations.Open($deckPath, $true, $false, $false)
    $presentation.SaveAs($pdfPath, 32)
    $presentation.Export($previewPath, 'PNG', 1600, 900)
    Write-Output "Exported $($presentation.Slides.Count) slides: $pdfPath"
} finally {
    if ($null -ne $presentation) {
        $presentation.Close()
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($presentation) | Out-Null
    }
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($application) | Out-Null
}
