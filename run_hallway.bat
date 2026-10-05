@echo off
title ARUGA - Hallway Launcher
cd /d "%~dp0"

echo ==========================================================
echo    ARUGA Hallway - Multi-Person Fall Monitor (Desktop)
echo    Self-contained: runs fully offline after this setup.
echo ==========================================================
echo.

:: 1. Check if virtual environment already exists
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment found.
    goto LAUNCH
)

:: 2. If .venv is missing, find a working Python installation on the system
echo [*] Virtual environment (.venv) not found. Checking system Python...

set PYTHON_CMD=
for %%C in (py python3 python) do (
    %%C --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=%%C
        goto PYTHON_FOUND
    )
)

:PYTHON_NOT_FOUND
echo.
echo [ERROR] Python is not installed or not found in your system PATH!
echo.
echo To run this project, the recipient needs to install Python (3.10 - 3.12 recommended):
echo   1. Download Python from: https://www.python.org/downloads/
echo   2. IMPORTANT: Check the box "Add Python to PATH" during installation.
echo   3. IMPORTANT: On the Optional Features step, keep "tcl/tk and IDLE" checked (needed for the desktop app).
echo.
pause
exit /b 1

:PYTHON_FOUND
echo [OK] Found system Python: %PYTHON_CMD%
echo.
echo [*] Setting up virtual environment for the first time...
echo [*] This will take a few minutes (ONNX Runtime + dependencies)...
echo [*] After this setup completes, the app runs 100%% offline.
echo.

%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create virtual environment (.venv).
    echo Please make sure your Python installation has the 'venv' module.
    pause
    exit /b 1
)

echo [*] Installing dependencies from requirements.txt...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install required packages. Please check your internet connection.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Environment setup complete! The app now runs fully offline.
echo.

:LAUNCH
echo [*] Starting ARUGA Hallway app...
echo [*] First start benchmarks CPU vs GPU backends (takes a few seconds)...
echo.
.\.venv\Scripts\python.exe hallway_app.py

pause
