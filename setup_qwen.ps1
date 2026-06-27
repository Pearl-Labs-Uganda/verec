# Qwen2.5-VL Setup Script for VEREC
# This script automates the Ollama model setup and backend configuration

Write-Host "=== Qwen2.5-VL Setup for VEREC ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Ollama is installed
Write-Host "[1/5] Checking Ollama installation..." -ForegroundColor Yellow
try {
    $ollamaVersion = ollama --version 2>&1
    Write-Host "✓ Ollama is installed: $ollamaVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Ollama is not installed!" -ForegroundColor Red
    Write-Host "Please install from: https://ollama.com/download/windows" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 2: Check for existing models
Write-Host "[2/5] Checking for Qwen models..." -ForegroundColor Yellow
$ollamaList = ollama list 2>&1 | Out-String

if ($ollamaList -match "qwen2\.5-vl:3b") {
    Write-Host "✓ qwen2.5-vl:3b is already available" -ForegroundColor Green
} elseif ($ollamaList -match "wen2\.5v1:3b") {
    Write-Host "Found wen2.5v1:3b, renaming to qwen2.5-vl:3b..." -ForegroundColor Yellow
    ollama cp wen2.5v1:3b qwen2.5-vl:3b
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Model renamed successfully" -ForegroundColor Green
        
        # Ask if user wants to delete the old tag
        $deleteOld = Read-Host "Delete old tag 'wen2.5v1:3b'? (y/N)"
        if ($deleteOld -eq "y" -or $deleteOld -eq "Y") {
            ollama rm wen2.5v1:3b
            Write-Host "✓ Old tag removed" -ForegroundColor Green
        }
    } else {
        Write-Host "✗ Failed to rename model" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "No Qwen model found. Pulling qwen2.5-vl:3b (~6GB download)..." -ForegroundColor Yellow
    Write-Host "This may take a few minutes depending on your connection..." -ForegroundColor Gray
    ollama pull qwen2.5-vl:3b
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Model downloaded successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to download model" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 3: Verify Ollama is running
Write-Host "[3/5] Verifying Ollama service..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
    Write-Host "✓ Ollama service is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Ollama service is not responding" -ForegroundColor Red
    Write-Host "Trying to start Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
        Write-Host "✓ Ollama service started" -ForegroundColor Green
    } catch {
        Write-Host "✗ Could not start Ollama. Please start it manually." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 4: Check/Create .env file
Write-Host "[4/5] Configuring environment..." -ForegroundColor Yellow
$envPath = ".env"
$envContent = ""

if (Test-Path $envPath) {
    $envContent = Get-Content $envPath -Raw
    Write-Host "Found existing .env file" -ForegroundColor Gray
    
    if ($envContent -match "USE_QWEN_VL") {
        # Update existing value
        $envContent = $envContent -replace "USE_QWEN_VL=.*", "USE_QWEN_VL=true"
        Write-Host "Updated USE_QWEN_VL=true in .env" -ForegroundColor Green
    } else {
        # Append new value
        $envContent += "`nUSE_QWEN_VL=true`n"
        Write-Host "Added USE_QWEN_VL=true to .env" -ForegroundColor Green
    }
    
    $envContent | Set-Content $envPath -NoNewline
} else {
    # Create new .env from template
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" $envPath
        $envContent = Get-Content $envPath -Raw
        $envContent = $envContent -replace "USE_QWEN_VL=false", "USE_QWEN_VL=true"
        $envContent | Set-Content $envPath -NoNewline
        Write-Host "✓ Created .env from template with USE_QWEN_VL=true" -ForegroundColor Green
    } else {
        # Create minimal .env
        "USE_QWEN_VL=true`nOLLAMA_URL=http://localhost:11434`n" | Set-Content $envPath
        Write-Host "✓ Created new .env with USE_QWEN_VL=true" -ForegroundColor Green
    }
}

Write-Host ""

# Step 5: Run test
Write-Host "[5/5] Running integration test..." -ForegroundColor Yellow
Write-Host "Testing Qwen connection..." -ForegroundColor Gray
python test_qwen.py --skip-check 2>&1 | Select-Object -First 20
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Test completed" -ForegroundColor Green
} else {
    Write-Host "⚠ Test had issues, but setup is complete" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Start the backend:" -ForegroundColor Gray
Write-Host "   python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3" -ForegroundColor White
Write-Host ""
Write-Host "2. Verify Qwen is active:" -ForegroundColor Gray
Write-Host "   curl http://localhost:8000/api/system" -ForegroundColor White
Write-Host "   (Look for 'vlm': 'Qwen2.5-VL-3B (Ollama)')" -ForegroundColor Gray
Write-Host ""
Write-Host "3. To disable Qwen and use local FastVLM:" -ForegroundColor Gray
Write-Host "   Set USE_QWEN_VL=false in .env" -ForegroundColor White
Write-Host ""
Write-Host "Documentation: See QWEN_SETUP.md for details" -ForegroundColor Gray
