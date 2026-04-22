# ─── Wikipedia dump download (Korean + English sample) ─────────────
# PowerShell:  .\data\download_wiki.ps1
# Requires:    Python + pip install wikiextractor

Push-Location $PSScriptRoot

# ─── Korean Wikipedia (~800 MB compressed) ────────────────────────
if (-not (Test-Path "kowiki.xml.bz2")) {
    Write-Host "Downloading Korean Wikipedia dump..."
    Invoke-WebRequest `
        -Uri "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2" `
        -OutFile "kowiki.xml.bz2"
}

# ─── English Wikipedia first chunk (~300 MB compressed) ───────────
if (-not (Test-Path "enwiki-p1.xml.bz2")) {
    Write-Host "Downloading English Wikipedia chunk 1..."
    Invoke-WebRequest `
        -Uri "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2" `
        -OutFile "enwiki-p1.xml.bz2"
}

# ─── Extract text ─────────────────────────────────────────────────
Write-Host "Extracting text (wikiextractor)..."
pip install wikiextractor --quiet
python -m wikiextractor.WikiExtractor kowiki.xml.bz2 -o extracted_ko --no-templates -q
python -m wikiextractor.WikiExtractor enwiki-p1.xml.bz2 -o extracted_en --no-templates -q

# ─── Build samples (first 1000 lines from first file) ─────────────
Write-Host "Building samples..."
Get-Content "extracted_ko\AA\wiki_00" -TotalCount 1000 -Encoding UTF8 |
    Out-File -Encoding UTF8 "sample_ko.txt"
Get-Content "extracted_en\AA\wiki_00" -TotalCount 1000 -Encoding UTF8 |
    Out-File -Encoding UTF8 "sample_en.txt"

Write-Host ""
Write-Host "=== Korean sample ==="
Get-Content "sample_ko.txt" -TotalCount 5
Write-Host ""
Write-Host "=== English sample ==="
Get-Content "sample_en.txt" -TotalCount 5

Pop-Location

Write-Host ""
Write-Host "Done. Run benchmarks:"
Write-Host "  ./build/test_wiki data/sample_ko.txt"
Write-Host "  ./build/test_wiki data/sample_en.txt"
Write-Host "  ./build/bench_perplexity data/sample_ko.txt"
