@echo off
REM Qwen2.5-VL Setup Script for VEREC (CMD version)

echo === Qwen2.5-VL Setup for VEREC ===
echo.

REM Check if Ollama is installed
echo [1/4] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed!
    echo Please install from: https://ollama.com/download/windows
    exit /b 1
)
echo [OK] Ollama is installed
echo.

REM Check and rename model if needed
echo [2/4] Setting up Qwen model...
ollama list | findstr "qwen2.5-vl:3b" >nul 2>&1
if not errorlevel 1 (
    echo [OK] qwen2.5-vl:3b is already available
) else (
    ollama list | findstr "wen2.5v1:3b" >nul 2>&1
    if not errorlevel 1 (
        echo Renaming wen2.5v1:3b to qwen2.5-vl:3b...
        ollama cp wen2.5v1:3b qwen2.5-vl:3b
        echo [OK] Model renamed
    ) else (
        echo Pulling qwen2.5-vl:3b (this may take a few minutes)...
        ollama pull qwen2.5-vl:3b
        echo [OK] Model downloaded
    )
)
echo.

REM Configure .env
echo [3/4] Configuring environment...
if exist .env (
    echo USE_QWEN_VL=true>> .env
    echo [OK] Updated .env
) else (
    echo USE_QWEN_VL=true> .env
    echo OLLAMA_URL=http://localhost:11434>> .env
    echo [OK] Created .env
)
echo.

REM Test
echo [4/4] Testing connection...
python test_qwen.py --skip-check
echo.

echo === Setup Complete! ===
echo.
echo Next: python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
echo.
pause
