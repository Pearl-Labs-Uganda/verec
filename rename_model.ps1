# Quick script to rename wen2.5v1:3b to qwen2.5-vl:3b

Write-Host "Renaming Ollama model..." -ForegroundColor Cyan

# Check if source exists
$models = ollama list 2>&1 | Out-String
if ($models -match "wen2\.5v1:3b") {
    Write-Host "Found wen2.5v1:3b" -ForegroundColor Green
    
    # Rename
    Write-Host "Copying to qwen2.5-vl:3b..." -ForegroundColor Yellow
    ollama cp wen2.5v1:3b qwen2.5-vl:3b
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Successfully renamed!" -ForegroundColor Green
        Write-Host ""
        
        # Verify
        Write-Host "Current models:" -ForegroundColor Cyan
        ollama list
        Write-Host ""
        
        # Offer to delete old
        $response = Read-Host "Delete old 'wen2.5v1:3b' tag? (y/N)"
        if ($response -eq "y" -or $response -eq "Y") {
            ollama rm wen2.5v1:3b
            Write-Host "✓ Old tag removed" -ForegroundColor Green
        } else {
            Write-Host "Keeping both tags (they point to the same model data)" -ForegroundColor Gray
        }
    } else {
        Write-Host "✗ Failed to rename model" -ForegroundColor Red
        exit 1
    }
} elseif ($models -match "qwen2\.5-vl:3b") {
    Write-Host "✓ qwen2.5-vl:3b already exists, nothing to do!" -ForegroundColor Green
} else {
    Write-Host "✗ Neither wen2.5v1:3b nor qwen2.5-vl:3b found" -ForegroundColor Red
    Write-Host "Run: ollama pull qwen2.5-vl:3b" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Next step: Enable Qwen in your backend" -ForegroundColor Cyan
Write-Host "  Set: " -NoNewline -ForegroundColor Gray
Write-Host '$env:USE_QWEN_VL="true"' -ForegroundColor White
