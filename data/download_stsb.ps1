# ─── STS-B benchmark download ─────────────────────────────────────
# PowerShell:  .\data\download_stsb.ps1
#
# Converts the official STS-B release into the simple 3-column TSV
# format expected by bench_stsb.c:
#
#     <score>\t<sentence1>\t<sentence2>
#
# Source: https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark

Push-Location $PSScriptRoot

if (-not (Test-Path "stsbenchmark")) {
    if (-not (Test-Path "stsb.tar.gz")) {
        Write-Host "Downloading STS-B..."
        Invoke-WebRequest `
            -Uri "http://ixa2.si.ehu.eus/stswiki/images/4/48/Stsbenchmark.tar.gz" `
            -OutFile "stsb.tar.gz"
    }
    Write-Host "Extracting..."
    tar -xzf stsb.tar.gz
}

# Source format: genre \t filename \t year \t id \t score \t s1 \t s2
# Combine train + dev + test
$in_files = @(
    "stsbenchmark\sts-train.csv",
    "stsbenchmark\sts-dev.csv",
    "stsbenchmark\sts-test.csv"
)
$out_file = "stsb.tsv"

if (Test-Path $out_file) { Remove-Item $out_file }

foreach ($f in $in_files) {
    if (-not (Test-Path $f)) { continue }
    Get-Content $f -Encoding UTF8 | ForEach-Object {
        $cols = $_ -split "`t"
        if ($cols.Length -ge 7) {
            $score = $cols[4]
            $s1    = $cols[5]
            $s2    = $cols[6]
            "$score`t$s1`t$s2" | Out-File -Append -Encoding UTF8 $out_file
        }
    }
}

$count = (Get-Content $out_file | Measure-Object -Line).Lines
Write-Host "Wrote $out_file ($count pairs)"
Write-Host ""
Write-Host "Sample:"
Get-Content $out_file -TotalCount 3

Pop-Location

Write-Host ""
Write-Host "Run benchmark:"
Write-Host "  ./build/bench_stsb data/stsb.tsv"
