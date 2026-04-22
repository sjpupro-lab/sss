# ─── Generate QA TSV from wiki text ─────────────────────────────
# Takes consecutive clause pairs and writes a QA TSV.
# Each pair (c_i, c_{i+1}) becomes one line: <c_i>\t<c_{i+1}>
#
# Usage:
#   .\data\make_qa.ps1 data\sample_ko.txt  [max_pairs]  > data\qa.tsv
#   .\data\make_qa.ps1 data\sample_en.txt  300          > data\qa_en.tsv

param(
    [Parameter(Mandatory=$true)] [string]$InputFile,
    [int]$MaxPairs = 500
)

if (-not (Test-Path $InputFile)) {
    Write-Error "File not found: $InputFile"
    exit 1
}

$lines = Get-Content $InputFile -Encoding UTF8
$clauses = New-Object System.Collections.Generic.List[string]

foreach ($line in $lines) {
    # Skip meta and blank lines
    if ($line -match '^\s*<' -or $line.Trim().Length -lt 10) { continue }

    # Split by sentence-ending punctuation
    $parts = $line -split '(?<=[\.\!\?])\s+'
    foreach ($p in $parts) {
        $t = $p.Trim()
        if ($t.Length -ge 10) { $clauses.Add($t) }
    }
}

# Emit consecutive pairs
$count = 0
for ($i = 0; $i -lt $clauses.Count - 1; $i++) {
    if ($count -ge $MaxPairs) { break }
    $q = $clauses[$i]     -replace '\t', ' '
    $a = $clauses[$i + 1] -replace '\t', ' '
    Write-Output "$q`t$a"
    $count++
}

Write-Error "Emitted $count QA pairs from $($clauses.Count) clauses."
